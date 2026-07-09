#!/usr/bin/env python3
"""
Recovery-rate robustness experiment for MSH under matched transmissibility.

Purpose
-------
Evaluate SIR final infection scale F(tc) when the recovery probability gamma varies.
Unlike a fixed-beta test, this script adjusts the per-step infection probability beta
so that the target edge transmissibility remains fixed at

    T_target = lambda_factor * beta_th.

For the discrete-time SIR update used here, infected nodes first attempt infection
and then recover with probability gamma. Under this rule,

    T(beta, gamma) = beta / (beta + gamma - beta * gamma),

so beta is obtained from

    beta = T_target * gamma / (1 - T_target * (1 - gamma)).

Default protocol
----------------
- gamma in {0.50, 0.75, 1.00}
- fixed seed ratio p = 5%
- lambda_factor = 2.5, i.e., T_target = 2.5 * beta_th
- 50 blocks x 20 SIR realizations per block
- Shared SIR random seeds across methods for paired comparisons
- MSH/HOSH is compared with the strongest non-MSH baseline under each setting
- Paired Wilcoxon signed-rank test with Benjamini-Hochberg correction

Outputs
-------
results/exp_sir_recovery_rate_robustness/
    RecoveryRateRobustness_Final_Statistical_Test_Summary.xlsx
    RecoveryRateRobustness_Checkpoint.xlsx
    RecoveryRateRobustness_Final_Stats_BH.csv
    RecoveryRateRobustness_Method_Summary.csv
    RecoveryRateRobustness_Raw_Blocks.csv
    RecoveryRate5pct_TableOnly_Final_Statistical_Test_Summary.xlsx
    RecoveryRate5pct_TableOnly_Checkpoint.xlsx
    RecoveryRate5pct_TableOnly_Final_Stats_BH.csv
    RecoveryRate5pct_TableOnly_Method_Summary.csv
    RecoveryRate5pct_TableOnly_Raw_Blocks.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import math
import os
import platform
import random
import sys
import traceback
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from tqdm import tqdm

from hosh_methods import get_node_scores  # type: ignore
METHOD_IMPLEMENTATION_SOURCE = "hosh_methods"

from network_loader import download_and_load_graph, get_network_list

try:
    from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes
except Exception:
    load_precomputed_rankings = None

    def get_standardized_ranked_nodes(scores: Dict, round_decimals: int = 8):
        return sorted(scores.keys(), key=lambda n: (-round(float(scores[n]), round_decimals), n))


# =============================================================================
# Plot configuration
# =============================================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})

DEFAULT_METHODS = [
    "HOSH", "VoteRank", "SNIM", "CHBC", "ISH", "DC", "BC", "CC", "K-Shell", "SH", "CI", "SNC"
]


# =============================================================================
# Reproducibility and statistics
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(*items: object, modulus: int = 1_000_000_000) -> int:
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % modulus


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def method_label(method: str, proposed_method: str, proposed_label: str) -> str:
    return proposed_label if method == proposed_method else method


def format_network_name(name: str) -> str:
    s = str(name)
    return s[:1].upper() + s[1:].lower() if s else s


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr)) if arr.size else np.nan
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci_val = float(t.ppf(0.975, df=arr.size - 1) * std_val / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean_val, std_val, ci_val


def fmt_mean_ci(mean_val: float, ci_val: float) -> str:
    if not np.isfinite(mean_val) or not np.isfinite(ci_val):
        return ""
    return f"{mean_val:.2f} ± {ci_val:.2f}"


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return q_values
    p = p_values[valid]
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)
    q_valid = np.empty_like(p)
    q_valid[order] = ranked_q
    q_values[valid] = q_valid
    return q_values


def sig_stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "*" if q < 0.05 else ""


# =============================================================================
# Threshold and transmissibility
# =============================================================================
def degree_moment_threshold(graph: nx.Graph) -> float:
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    if degrees.size == 0:
        return np.nan
    k1 = float(np.mean(degrees))
    k2 = float(np.mean(degrees ** 2))
    denom = k2 - k1
    if denom <= 0:
        return np.nan
    return k1 / denom


def beta_from_transmissibility(T: float, gamma: float) -> float:
    """Invert T(beta,gamma)=beta/(beta+gamma-beta*gamma)."""
    denom = 1.0 - T * (1.0 - gamma)
    if denom <= 0:
        return np.nan
    return (T * gamma) / denom


# =============================================================================
# SIR model
# =============================================================================
def run_sir_final(graph: nx.Graph, seeds: Iterable, beta: float, gamma: float,
                  rng_seed: int, max_steps: int = 1000) -> int:
    """Discrete-time synchronous SIR. Returns final infected + recovered count."""
    rng = random.Random(rng_seed)
    infected = set(n for n in seeds if graph.has_node(n))
    recovered = set()

    if not infected:
        return 0

    for _ in range(max_steps):
        if not infected:
            break

        new_infected = set()
        new_recovered = set()

        for u in list(infected):
            for v in graph.neighbors(u):
                if v not in infected and v not in recovered:
                    if rng.random() < beta:
                        new_infected.add(v)
            if rng.random() < gamma:
                new_recovered.add(u)

        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)

    return len(infected) + len(recovered)


def shared_sir_seed(network_name: str, seed_budget_key: str, gamma: float, beta: float,
                    block_idx: int, repeat_idx: int, master_seed: int) -> int:
    # Method name is intentionally excluded so paired comparisons use common random numbers.
    return int((master_seed + stable_int_hash(
        "sir_recovery", network_name, seed_budget_key, f"gamma={gamma:.6f}", f"beta={beta:.8f}", block_idx, repeat_idx
    )) % (2**32 - 1))


def evaluate_blocks(graph: nx.Graph, seeds: List, beta: float, gamma: float,
                    network_name: str, seed_budget_key: str,
                    blocks: int, repeats: int, master_seed: int,
                    max_steps: int) -> np.ndarray:
    n = graph.number_of_nodes()
    vals = []
    for b in range(blocks):
        total = 0
        for r in range(repeats):
            rng_seed = shared_sir_seed(network_name, seed_budget_key, gamma, beta, b, r, master_seed)
            total += run_sir_final(graph, seeds, beta, gamma, rng_seed=rng_seed, max_steps=max_steps)
        vals.append((total / repeats) / n * 100.0)
    return np.asarray(vals, dtype=float)


# =============================================================================
# Ranking computation
# =============================================================================
def load_or_compute_scores(method: str, graph: nx.Graph, network_name: str,
                           use_precomputed: bool) -> Dict:
    if use_precomputed and load_precomputed_rankings is not None:
        rankings = load_precomputed_rankings(network_name)
        if rankings and method in rankings and rankings[method] is not None:
            return rankings[method]
        if rankings and method == "HOSH-Sqrt" and "HOSH-BoxCox" in rankings:
            return rankings["HOSH-BoxCox"]
    return get_node_scores(method, graph)


def compute_rankings_once(methods: List[str], graph: nx.Graph, network_name: str,
                          use_precomputed: bool) -> Dict[str, Dict]:
    rankings = {}
    for m in methods:
        print(f"    Ranking: {m}")
        rankings[m] = load_or_compute_scores(m, graph, network_name, use_precomputed)
    return rankings


def select_top_k(scores: Dict, k: int) -> List:
    ranked = get_standardized_ranked_nodes(scores, round_decimals=8)
    return ranked[:k]


# =============================================================================
# Export and visualization
# =============================================================================
def collect_environment_metadata(args, notes: str) -> pd.DataFrame:
    rows = [
        ("Generated at", _dt.datetime.now().isoformat(timespec="seconds")),
        ("Experiment", "Recovery-rate SIR robustness under matched transmissibility"),
        ("Protocol", notes),
        ("Gamma values", args.gammas),
        ("Seed ratio", args.seed_ratios),
        ("Lambda factor", str(args.lambda_factor)),
        ("Blocks", str(args.blocks)),
        ("Repeats per block", str(args.repeats)),
        ("Shared random seeds", "Yes; common block/repeat SIR seeds are used across methods under each network-setting."),
        ("Use precomputed rankings", str(args.use_precomputed)),
        ("Method implementation source", METHOD_IMPLEMENTATION_SOURCE),
        ("Proposed method", args.proposed_method),
        ("Proposed label", args.proposed_label),
        ("Master seed", str(args.master_seed)),
        ("Python version", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("NetworkX version", getattr(nx, "__version__", "unknown")),
        ("NumPy version", getattr(np, "__version__", "unknown")),
        ("Pandas version", getattr(pd, "__version__", "unknown")),
        ("Matplotlib version", getattr(matplotlib, "__version__", "unknown")),
    ]
    try:
        import scipy
        rows.append(("SciPy version", getattr(scipy, "__version__", "unknown")))
    except Exception:
        pass
    return pd.DataFrame(rows, columns=["Item", "Value"])


def make_output_prefix() -> str:
    return "RecoveryRate5pct_TableOnly"


def save_outputs(output_dir: str, final_records: List[Dict], method_records: List[Dict], raw_records: List[Dict],
                 ranking_records: List[Dict], args, notes: str, final_name: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    final_df = pd.DataFrame(final_records)
    method_df = pd.DataFrame(method_records)
    raw_df = pd.DataFrame(raw_records)
    ranking_df = pd.DataFrame(ranking_records)

    if not final_df.empty:
        final_df["Q_Value_BH"] = benjamini_hochberg(final_df["P_Value_Final_vs_Best"].to_numpy(dtype=float))
        final_df["Significant_0.05_BH"] = final_df["Q_Value_BH"] < 0.05
        final_df["Significance"] = final_df["Q_Value_BH"].map(sig_stars)

    summary_by_gamma_seed = pd.DataFrame()
    if not final_df.empty:
        summary_by_gamma_seed = (
            final_df.groupby(["Gamma", "Seed_Ratio_Label"])
            .agg(
                Settings=("Network", "count"),
                Mean_Improvement_pct=("Improvement(%)", "mean"),
                Median_Improvement_pct=("Improvement(%)", "median"),
                Mean_DeltaF_pct=("Delta_Final(%)", "mean"),
                Mean_WinRate_pct=("Win_Rate_Blocks(%)", "mean"),
                Significant_Count_BH=("Significant_0.05_BH", "sum"),
            )
            .reset_index()
        )

    # Manuscript-ready detail table: one row per network-gamma setting.
    table_s5_detail = pd.DataFrame()
    table_s5_summary = pd.DataFrame()
    if not final_df.empty:
        table_s5_detail = final_df[[
            "Network", "Gamma", "Beta_th", "Target_Transmissibility", "Beta_Adjusted",
            "MSH_Final_Mean±CI", "Best_Baseline_Label", "Best_Final_Mean±CI",
            "Delta_Final(%)", "Improvement(%)", "Q_Value_BH", "Significance"
        ]].copy()
        table_s5_detail.rename(columns={
            "Beta_th": "beta_th",
            "Target_Transmissibility": "T_target",
            "Beta_Adjusted": "beta_adjusted",
            "Best_Baseline_Label": "Best baseline",
            "Delta_Final(%)": "Delta F(%)",
            "Improvement(%)": "Improvement eta(%)",
            "Q_Value_BH": "p_BH",
        }, inplace=True)
        table_s5_summary = (
            final_df.groupby("Gamma")
            .agg(
                Settings=("Network", "count"),
                Mean_Improvement_pct=("Improvement(%)", "mean"),
                Median_Improvement_pct=("Improvement(%)", "median"),
                Mean_DeltaF_pct=("Delta_Final(%)", "mean"),
                Mean_WinRate_pct=("Win_Rate_Blocks(%)", "mean"),
                Significant_Count_BH=("Significant_0.05_BH", "sum"),
            )
            .reset_index()
        )

    xlsx_path = os.path.join(output_dir, final_name)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table_s5_summary.to_excel(writer, sheet_name="Table_S5_Summary", index=False)
        table_s5_detail.to_excel(writer, sheet_name="Table_S5_Detail", index=False)
        final_df.to_excel(writer, sheet_name="Final_Stats_BH", index=False)
        summary_by_gamma_seed.to_excel(writer, sheet_name="Summary_By_Gamma_Seed", index=False)
        method_df.to_excel(writer, sheet_name="Method_FinalSpread", index=False)
        raw_df.to_excel(writer, sheet_name="Raw_Blocks", index=False)
        ranking_df.to_excel(writer, sheet_name="Rankings", index=False)
        collect_environment_metadata(args, notes).to_excel(writer, sheet_name="Protocol_Metadata", index=False)
        pd.DataFrame({"Notes": [notes]}).to_excel(writer, sheet_name="Notes", index=False)

    prefix = make_output_prefix()
    final_df.to_csv(os.path.join(output_dir, f"{prefix}_Final_Stats_BH.csv"), index=False, encoding="utf-8-sig")
    method_df.to_csv(os.path.join(output_dir, f"{prefix}_Method_Summary.csv"), index=False, encoding="utf-8-sig")
    raw_df.to_csv(os.path.join(output_dir, f"{prefix}_Raw_Blocks.csv"), index=False, encoding="utf-8-sig")
    if 'table_s5_summary' in locals() and not table_s5_summary.empty:
        table_s5_summary.to_csv(os.path.join(output_dir, f"{prefix}_Table_S5_Summary.csv"), index=False, encoding="utf-8-sig")
    if 'table_s5_detail' in locals() and not table_s5_detail.empty:
        table_s5_detail.to_csv(os.path.join(output_dir, f"{prefix}_Table_S5_Detail.csv"), index=False, encoding="utf-8-sig")

    print(f"    [Output] Workbook saved: {xlsx_path}")
    return final_df


def style_axes(ax) -> None:
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#000000")
    ax.tick_params(direction="out", which="major", length=3.0, width=0.7)


def plot_mean_improvement_heatmap(final_df: pd.DataFrame, output_dir: str) -> None:
    if final_df is None or final_df.empty:
        return
    table = final_df.pivot_table(index="Gamma", columns="Seed_Ratio_Label", values="Improvement(%)", aggfunc="mean")
    if table.empty:
        return
    # Keep seed-ratio columns in numeric order when possible.
    ordered_cols = sorted(table.columns, key=lambda x: float(str(x).replace("%", "")))
    table = table[ordered_cols]
    mat = table.to_numpy(dtype=float)
    finite_vals = mat[np.isfinite(mat)]
    if finite_vals.size == 0:
        return

    vmax = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([f"{g:.2f}" for g in table.index])
    ax.set_xlabel("Seed ratio")
    ax.set_ylabel(r"Recovery probability $\gamma$")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(r"Mean $\eta$ (%)")
    style_axes(ax)
    plt.tight_layout(pad=0.3)
    base = os.path.join(output_dir, f"{make_output_prefix()}_MeanImprovement_Heatmap")
    plt.savefig(base + ".pdf", format="pdf")
    plt.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"    [Output] Heatmap saved: {base}.pdf")


def plot_summary_table(final_df: pd.DataFrame, output_dir: str) -> None:
    if final_df is None or final_df.empty:
        return
    table = (
        final_df.groupby(["Gamma", "Seed_Ratio_Label"])
        .agg(
            Mean_Eta=("Improvement(%)", "mean"),
            Median_Eta=("Improvement(%)", "median"),
            Mean_DeltaF=("Delta_Final(%)", "mean"),
            Mean_Win=("Win_Rate_Blocks(%)", "mean"),
            Sig_BH=("Significant_0.05_BH", "sum"),
            Settings=("Network", "count"),
        )
        .reset_index()
    )
    display = table.copy()
    display["Gamma"] = display["Gamma"].map(lambda x: f"{x:.2f}")
    for col in ["Mean_Eta", "Median_Eta", "Mean_DeltaF", "Mean_Win"]:
        display[col] = display[col].map(lambda x: f"{x:.2f}")
    display["Significant"] = display.apply(lambda r: f"{int(r['Sig_BH'])}/{int(r['Settings'])}", axis=1)
    display = display[["Gamma", "Seed_Ratio_Label", "Mean_Eta", "Median_Eta", "Mean_DeltaF", "Mean_Win", "Significant"]]
    display.columns = [r"$\gamma$", "Seed ratio", r"Mean $\eta$", r"Median $\eta$", r"Mean $\Delta F$", "Mean win (%)", "BH sig."]

    fig_height = max(2.0, 0.34 * len(display) + 0.65)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    ax.axis("off")
    tab = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center", colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.18)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    plt.tight_layout(pad=0.2)
    out_pdf = os.path.join(output_dir, f"{make_output_prefix()}_Summary_Table.pdf")
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"    [Output] Summary table saved: {out_pdf}")


# =============================================================================
# Main experiment
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery-rate SIR robustness under matched transmissibility.")
    parser.add_argument("--output-dir", default="results/exp_sir_recovery_rate_5pct_table_only")
    parser.add_argument("--networks", default="", help="Comma-separated networks; empty means get_network_list().")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--gammas", default="0.5,0.75,1.0")
    parser.add_argument("--seed-ratios", default="0.05", help="Fixed seed ratio. Default is 0.05 (5%).")
    parser.add_argument("--lambda-factor", type=float, default=2.5)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.set_defaults(use_precomputed=True)
    parser.add_argument("--fresh-rankings", dest="use_precomputed", action="store_false",
                        help="Recompute rankings from hosh_methods.py instead of using precomputed ranking files.")
    parser.add_argument("--proposed-method", default="HOSH")
    parser.add_argument("--proposed-label", default="MSH")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    networks = parse_csv_list(args.networks) if args.networks else get_network_list()
    methods = parse_csv_list(args.methods)
    gammas = parse_float_list(args.gammas)
    seed_ratios = parse_float_list(args.seed_ratios)
    if args.proposed_method not in methods:
        methods = [args.proposed_method] + methods

    notes = (
        "Recovery-rate robustness experiment with fixed seed ratio p=5%. For each network, the target transmissibility is "
        "T=lambda_factor*beta_th. The per-step infection probability beta is adjusted for each gamma by "
        "beta=T*gamma/[1-T*(1-gamma)]. Final infection scale is evaluated using block-level common random numbers. "
        "MSH/HOSH is compared with the strongest non-MSH baseline using a paired Wilcoxon signed-rank test; "
        "p-values are adjusted by Benjamini-Hochberg correction across all network-gamma settings."
    )

    print("=" * 80)
    print("Recovery-rate SIR robustness under matched transmissibility (5% seed ratio, table only)")
    print("=" * 80)
    print(f"Networks: {networks}")
    print(f"Methods: {methods}")
    print(f"Gammas: {gammas}")
    print(f"Seed ratios: {seed_ratios}")
    print(f"lambda factor={args.lambda_factor}, blocks={args.blocks}, repeats={args.repeats}")
    print(f"Method implementation source: {METHOD_IMPLEMENTATION_SOURCE}")
    print(f"Use precomputed rankings: {args.use_precomputed}")
    print("=" * 80)

    set_seed(args.master_seed)

    final_records: List[Dict] = []
    method_records: List[Dict] = []
    raw_records: List[Dict] = []
    ranking_records: List[Dict] = []

    checkpoint_name = f"{make_output_prefix()}_Checkpoint.xlsx"
    final_name = f"{make_output_prefix()}_Final_Statistical_Test_Summary.xlsx"

    for net_idx, net in enumerate(networks, 1):
        print(f"\n[{net_idx}/{len(networks)}] Network: {net}")
        print("-" * 80)
        try:
            graph = download_and_load_graph(net, verbose=False)
            if graph is None or graph.number_of_nodes() == 0:
                print(f"  [Skip] {net}: failed to load or empty graph.")
                continue

            n = graph.number_of_nodes()
            e = graph.number_of_edges()
            net_label = format_network_name(net)
            beta_th = degree_moment_threshold(graph)
            if not np.isfinite(beta_th):
                print(f"  [Skip] {net}: invalid beta_th.")
                continue
            T_target = args.lambda_factor * beta_th
            if T_target >= 1.0:
                print(f"  [Skip] {net}: T_target={T_target:.6f} >= 1. Lower --lambda-factor.")
                continue
            print(f"  Nodes={n}, Edges={e}, beta_th={beta_th:.6f}, T_target={T_target:.6f}")

            rankings = compute_rankings_once(methods, graph, net, args.use_precomputed)

            for seed_ratio in seed_ratios:
                k = max(1, int(n * seed_ratio))
                seed_ratio_label = f"{seed_ratio * 100:.0f}%" if abs(seed_ratio * 100 - round(seed_ratio * 100)) < 1e-9 else f"{seed_ratio * 100:.1f}%"
                seed_budget_key = f"p={seed_ratio:.6f};k={k}"
                seed_sets = {}
                for m in methods:
                    seeds = select_top_k(rankings[m], k)
                    seed_sets[m] = seeds
                    ranking_records.append({
                        "Network": net_label,
                        "Method": m,
                        "MethodLabel": method_label(m, args.proposed_method, args.proposed_label),
                        "Seed_Ratio": seed_ratio,
                        "Seed_Ratio_Label": seed_ratio_label,
                        "k": k,
                        "TopK_Seeds": ",".join(map(str, seeds)),
                        "UsedPrecomputed": args.use_precomputed,
                    })

                for gamma in gammas:
                    beta = beta_from_transmissibility(T_target, gamma)
                    if not np.isfinite(beta) or beta < 0 or beta > 1:
                        print(f"  [Skip] gamma={gamma:.3f}, seed={seed_ratio_label}: invalid beta={beta}")
                        continue
                    print(f"  [Setting] seed={seed_ratio_label}, gamma={gamma:.3f}, beta={beta:.6f}")
                    method_blocks = {}

                    for m in tqdm(methods, desc=f"    p={seed_ratio_label}, gamma={gamma:.2f}", leave=False):
                        blocks = evaluate_blocks(
                            graph, seed_sets[m], beta=beta, gamma=gamma,
                            network_name=net,
                            seed_budget_key=seed_budget_key,
                            blocks=args.blocks, repeats=args.repeats,
                            master_seed=args.master_seed, max_steps=args.max_steps,
                        )
                        method_blocks[m] = blocks
                        mean_val, std_val, ci_val = mean_ci95(blocks)
                        method_records.append({
                            "Network": net_label,
                            "N": n,
                            "E": e,
                            "Beta_th": beta_th,
                            "Lambda_Factor": args.lambda_factor,
                            "Target_Transmissibility": T_target,
                            "Beta_Adjusted": beta,
                            "Gamma": gamma,
                            "Seed_Ratio": seed_ratio,
                            "Seed_Ratio_Label": seed_ratio_label,
                            "k": k,
                            "Method": m,
                            "MethodLabel": method_label(m, args.proposed_method, args.proposed_label),
                            "Final_Mean(%)": mean_val,
                            "Final_Std": std_val,
                            "Final_95%_CI": ci_val,
                            "Final_Mean±CI": fmt_mean_ci(mean_val, ci_val),
                        })
                        for b_idx, val in enumerate(blocks):
                            raw_records.append({
                                "Network": net_label,
                                "Beta_th": beta_th,
                                "Lambda_Factor": args.lambda_factor,
                                "Target_Transmissibility": T_target,
                                "Beta_Adjusted": beta,
                                "Gamma": gamma,
                                "Seed_Ratio": seed_ratio,
                                "Seed_Ratio_Label": seed_ratio_label,
                                "k": k,
                                "Method": m,
                                "Block": b_idx,
                                "Final_Infection(%)": val,
                            })

                    if args.proposed_method not in method_blocks:
                        continue
                    msh_blocks = method_blocks[args.proposed_method]
                    msh_mean, msh_std, msh_ci = mean_ci95(msh_blocks)

                    best_name = None
                    best_mean = -np.inf
                    for m, vals in method_blocks.items():
                        if m == args.proposed_method:
                            continue
                        m_mean = float(np.mean(vals))
                        if m_mean > best_mean:
                            best_mean = m_mean
                            best_name = m
                    if best_name is None:
                        continue

                    best_blocks = method_blocks[best_name]
                    best_mean, best_std, best_ci = mean_ci95(best_blocks)
                    diff_blocks = msh_blocks - best_blocks
                    delta_mean = float(np.mean(diff_blocks))
                    diff_std = float(np.std(diff_blocks, ddof=1)) if len(diff_blocks) > 1 else 0.0
                    dz = delta_mean / diff_std if diff_std > 0 else np.nan
                    win_rate = float(np.mean(diff_blocks > 0) * 100.0)
                    improvement = (msh_mean - best_mean) / best_mean * 100.0 if best_mean > 0 else np.nan

                    try:
                        _, p_val = wilcoxon(msh_blocks, best_blocks, alternative="two-sided")
                        p_val = max(float(p_val), 1e-20)
                    except ValueError:
                        p_val = 1.0

                    final_records.append({
                        "Network": net_label,
                        "N": n,
                        "E": e,
                        "Beta_th": beta_th,
                        "Lambda_Factor": args.lambda_factor,
                        "Target_Transmissibility": T_target,
                        "Beta_Adjusted": beta,
                        "Gamma": gamma,
                        "Seed_Ratio": seed_ratio,
                        "Seed_Ratio_Label": seed_ratio_label,
                        "k": k,
                        "Best_Baseline_Final": best_name,
                        "Best_Baseline_Label": method_label(best_name, args.proposed_method, args.proposed_label),
                        "MSH_Final(%)": msh_mean,
                        "MSH_95%_CI": msh_ci,
                        "MSH_Final_Mean±CI": fmt_mean_ci(msh_mean, msh_ci),
                        "Best_Final(%)": best_mean,
                        "Best_95%_CI": best_ci,
                        "Best_Final_Mean±CI": fmt_mean_ci(best_mean, best_ci),
                        "Delta_Final(%)": msh_mean - best_mean,
                        "Mean_Paired_Difference(%)": delta_mean,
                        "Improvement(%)": improvement,
                        "Paired_Effect_dz": dz,
                        "Win_Rate_Blocks(%)": win_rate,
                        "P_Value_Final_vs_Best": p_val,
                    })
                    print(
                        f"    Best={best_name}; MSH={msh_mean:.2f}±{msh_ci:.2f}, "
                        f"Best={best_mean:.2f}±{best_ci:.2f}, improvement={improvement:.2f}%, p={p_val:.2e}"
                    )

                    save_outputs(args.output_dir, final_records, method_records, raw_records,
                                 ranking_records, args, notes, checkpoint_name)
                    gc.collect()

        except Exception as exc:
            print(f"  [Error] Failed on network {net}: {exc}")
            traceback.print_exc()
            continue

    save_outputs(args.output_dir, final_records, method_records, raw_records,
                 ranking_records, args, notes, final_name)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Final workbook: {os.path.join(args.output_dir, final_name)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
