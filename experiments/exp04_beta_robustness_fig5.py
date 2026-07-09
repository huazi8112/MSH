#!/usr/bin/env python3
"""
Fixed-beta SIR robustness experiment with statistical tests.

Purpose
-------
Evaluate the final infection scale F(tc) under a fixed infection-probability
range beta = 0.100, 0.125, 0.150, 0.175, 0.200, using a configurable seed budget.
The experiment reports MSH/HOSH against the strongest baseline at each
network-beta setting with paired statistical tests.

Default protocol
----------------
- Default seed budget: 5% of nodes; set --seed-count 1 for the single-node diagnostic
- Fixed infection probabilities: beta in {0.100, 0.125, 0.150, 0.175, 0.200}
- Recovery probability: gamma = 1.0
- 50 blocks x 20 SIR realizations per block
- Shared SIR random seeds across methods for paired comparisons
- Paired Wilcoxon signed-rank test: MSH vs best baseline
- Benjamini-Hochberg correction across all network-beta tests
- Mean +/- 95% CI, paired effect size dz, and block win rate

Important notes
---------------
- This is a fixed-beta sensitivity experiment, not the main threshold-multiplier SIR test.
- No Python built-in hash() is used for random seed derivation.
- By default, the script uses existing precomputed rankings when available, consistent with the non-runtime experiments.
  Use --fresh-rankings if you want to recompute rankings from the active hosh_methods.py.
- HOSH is labeled as MSH in figures/tables by default.

Outputs
-------
results/exp_sir_fixed_beta_5pct_stats/
    FixedBeta5pct_Final_Statistical_Test_Summary.xlsx
    FixedBeta5pct_Checkpoint.xlsx
    FixedBeta5pct_Improvement_Heatmap.pdf/png
    FixedBeta5pct_Statistical_Summary_Table.pdf
    FixedBeta5pct_Final_Stats_BH.csv
    FixedBeta5pct_Method_Summary.csv
    FixedBeta5pct_Raw_Blocks.csv
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from tqdm import tqdm

# Use the active project implementation.
# If you replaced hosh_methods.py with the optimized exact MSH/HOSH version, all non-runtime
# experiments will use that implementation whenever rankings need to be recomputed.
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
# 0. Plot configuration consistent with revised experiments
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

COLORS = {
    "MSH": "#D63230",
    "HOSH": "#D63230",
    "VoteRank": "#2CA02C",
    "SNIM": "#7F7F7F",
    "CHBC": "#5D3FD3",
    "ISH": "#F08C3D",
    "DC": "#E5B25D",
    "BC": "#4FA3D1",
    "CC": "#4364B8",
    "K-Shell": "#A855A8",
    "SH": "#E2739F",
    "CI": "#8D6E63",
    "SNC": "#4DB6AC",
}


# =============================================================================
# 1. Reproducibility and statistics
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
    """Display network names with only the first character capitalized."""
    s = str(name)
    return s[:1].upper() + s[1:].lower() if s else s


def make_output_prefix(args) -> str:
    if getattr(args, "budget_label", ""):
        return f"FixedBeta{args.budget_label}"
    if getattr(args, "seed_count", 0) and args.seed_count > 0:
        return f"FixedBetaTop{args.seed_count}"
    return f"FixedBeta{int(round(args.seed_ratio * 100))}pct"


def seed_budget_description(args, k: int) -> str:
    if getattr(args, "seed_count", 0) and args.seed_count > 0:
        return f"Top-{k}"
    return f"{args.seed_ratio:.1%}"


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
    """Return a single significance marker after BH correction.

    The manuscript uses only one symbol: * indicates q < 0.05;
    otherwise no marker is shown. This avoids multi-level significance
    annotations such as ** or ***.
    """
    if not np.isfinite(q):
        return ""
    return "*" if q < 0.05 else ""


# =============================================================================
# 2. SIR model
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


def shared_sir_seed(network_name: str, beta: float, method_setting_key: str,
                    block_idx: int, repeat_idx: int, master_seed: int) -> int:
    # method_setting_key intentionally does not include method name, so all methods share the same SIR seed.
    return int((master_seed + stable_int_hash("sir", network_name, f"{beta:.6f}", method_setting_key, block_idx, repeat_idx)) % (2**32 - 1))


def evaluate_blocks(graph: nx.Graph, seeds: List, beta: float, gamma: float,
                    network_name: str, method_setting_key: str,
                    blocks: int, repeats: int, master_seed: int,
                    max_steps: int) -> np.ndarray:
    n = graph.number_of_nodes()
    vals = []
    for b in range(blocks):
        total = 0
        for r in range(repeats):
            rng_seed = shared_sir_seed(network_name, beta, method_setting_key, b, r, master_seed)
            total += run_sir_final(graph, seeds, beta, gamma, rng_seed=rng_seed, max_steps=max_steps)
        vals.append((total / repeats) / n * 100.0)
    return np.asarray(vals, dtype=float)


# =============================================================================
# 3. Ranking computation
# =============================================================================
def load_or_compute_scores(method: str, graph: nx.Graph, network_name: str,
                           use_precomputed: bool) -> Dict:
    if use_precomputed and load_precomputed_rankings is not None:
        rankings = load_precomputed_rankings(network_name)
        if rankings and method in rankings and rankings[method] is not None:
            return rankings[method]
        if rankings and method == "HOSH-Sqrt" and "HOSH-BoxCox" in rankings:
            return rankings["HOSH-BoxCox"]
    # Fresh computation avoids stale precomputed rankings.
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
# 4. Export and visualization
# =============================================================================
def collect_environment_metadata(args, notes: str) -> pd.DataFrame:
    rows = [
        ("Generated at", _dt.datetime.now().isoformat(timespec="seconds")),
        ("Experiment", "Fixed-beta SIR final infection scale"),
        ("Protocol", notes),
        ("Beta values", args.betas),
        ("Gamma", str(args.gamma)),
        ("Seed ratio", str(args.seed_ratio)),
        ("Seed count", str(args.seed_count)),
        ("Budget label", str(args.budget_label)),
        ("Blocks", str(args.blocks)),
        ("Repeats per block", str(args.repeats)),
        ("Shared random seeds", "Yes; same block/repeat SIR seeds are used across methods under each network-beta setting."),
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

    summary_by_beta = pd.DataFrame()
    if not final_df.empty:
        summary_by_beta = (
            final_df.groupby("Beta")
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
        final_df.to_excel(writer, sheet_name="Final_Stats_BH", index=False)
        summary_by_beta.to_excel(writer, sheet_name="Summary_By_Beta", index=False)
        method_df.to_excel(writer, sheet_name="Method_FinalSpread", index=False)
        raw_df.to_excel(writer, sheet_name="Raw_Blocks", index=False)
        ranking_df.to_excel(writer, sheet_name="Rankings", index=False)
        collect_environment_metadata(args, notes).to_excel(writer, sheet_name="Protocol_Metadata", index=False)
        pd.DataFrame({"Notes": [notes]}).to_excel(writer, sheet_name="Notes", index=False)

    # CSV copies for quick inspection.
    final_df.to_csv(os.path.join(output_dir, f"{make_output_prefix(args)}_Final_Stats_BH.csv"), index=False, encoding="utf-8-sig")
    method_df.to_csv(os.path.join(output_dir, f"{make_output_prefix(args)}_Method_Summary.csv"), index=False, encoding="utf-8-sig")
    raw_df.to_csv(os.path.join(output_dir, f"{make_output_prefix(args)}_Raw_Blocks.csv"), index=False, encoding="utf-8-sig")

    print(f"    [Output] Workbook saved: {xlsx_path}")
    return final_df


def style_axes(ax) -> None:
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#000000")
    ax.tick_params(direction="out", which="major", length=3.0, width=0.7)


def plot_eta_heatmap(final_df: pd.DataFrame, output_dir: str, args) -> None:
    if final_df is None or final_df.empty:
        return
    betas = sorted(final_df["Beta"].dropna().unique())
    networks = list(final_df["Network"].drop_duplicates())
    if not betas or not networks:
        return

    mat = np.full((len(networks), len(betas)), np.nan, dtype=float)
    qmat = np.full_like(mat, np.nan, dtype=float)
    for i, net in enumerate(networks):
        for j, beta in enumerate(betas):
            sub = final_df[(final_df["Network"] == net) & (np.isclose(final_df["Beta"], beta))]
            if not sub.empty:
                mat[i, j] = float(sub.iloc[0]["Improvement(%)"])
                qmat[i, j] = float(sub.iloc[0]["Q_Value_BH"])

    finite_vals = mat[np.isfinite(mat)]
    if finite_vals.size == 0:
        return
    vmax = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_width = max(4.2, 0.75 * len(betas) + 2.2)
    fig_height = max(3.2, 0.34 * len(networks) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(np.arange(len(betas)))
    ax.set_xticklabels([f"{b:.3f}".rstrip("0").rstrip(".") for b in betas])
    ax.set_yticks(np.arange(len(networks)))
    ax.set_yticklabels([str(n) for n in networks])
    ax.set_xlabel(r"Infection probability $\beta$")
    ax.set_ylabel("Network")

    for i in range(len(networks)):
        for j in range(len(betas)):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            star = sig_stars(qmat[i, j])
            text = f"{val:.1f}{star}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(r"$\eta$ (%)")
    style_axes(ax)
    plt.tight_layout(pad=0.3)
    base = os.path.join(output_dir, f"{make_output_prefix(args)}_Improvement_Heatmap")
    plt.savefig(base + ".pdf", format="pdf")
    plt.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"    [Output] Heatmap saved: {base}.pdf")


def plot_summary_table(final_df: pd.DataFrame, output_dir: str, args) -> None:
    if final_df is None or final_df.empty:
        return
    table = (
        final_df.groupby("Beta")
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
    display["Beta"] = display["Beta"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip("."))
    for col in ["Mean_Eta", "Median_Eta", "Mean_DeltaF", "Mean_Win"]:
        display[col] = display[col].map(lambda x: f"{x:.2f}")
    display["Significant"] = display.apply(lambda r: f"{int(r['Sig_BH'])}/{int(r['Settings'])}", axis=1)
    display = display[["Beta", "Mean_Eta", "Median_Eta", "Mean_DeltaF", "Mean_Win", "Significant"]]
    display.columns = [r"$\beta$", r"Mean $\eta$", r"Median $\eta$", r"Mean $\Delta F$", "Mean win (%)", "BH sig."]

    fig_height = max(1.5, 0.34 * len(display) + 0.65)
    fig, ax = plt.subplots(figsize=(7.6, fig_height))
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
    out_pdf = os.path.join(output_dir, f"{make_output_prefix(args)}_Statistical_Summary_Table.pdf")
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"    [Output] Summary table saved: {out_pdf}")


# =============================================================================
# 5. Main experiment
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-beta SIR final infection scale with statistical tests.")
    parser.add_argument("--output-dir", default="results/exp_sir_fixed_beta_5pct_stats")
    parser.add_argument("--networks", default="", help="Comma-separated networks; empty means get_network_list().")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--betas", default="0.1,0.125,0.15,0.175,0.2")
    parser.add_argument("--seed-ratio", type=float, default=0.05)
    parser.add_argument("--seed-count", type=int, default=0, help="Fixed seed count. Use 1 for the single-node diagnostic. If >0, this overrides --seed-ratio.")
    parser.add_argument("--budget-label", default="5pct", help="Label used in output filenames, e.g., 5pct or SingleNode.")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.set_defaults(use_precomputed=True)
    parser.add_argument("--fresh-rankings", dest="use_precomputed", action="store_false",
                        help="Recompute rankings from the active hosh_methods.py instead of using precomputed ranking files.")
    parser.add_argument("--proposed-method", default="HOSH")
    parser.add_argument("--proposed-label", default="MSH")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    networks = parse_csv_list(args.networks) if args.networks else get_network_list()
    methods = parse_csv_list(args.methods)
    betas = parse_float_list(args.betas)
    if args.proposed_method not in methods:
        methods = [args.proposed_method] + methods

    notes = (
        "Fixed infection-probability SIR sensitivity experiment. "
        "For each network and beta, final infection scale is evaluated using block-level common random numbers. "
        "MSH/HOSH is compared with the strongest non-MSH baseline using a paired Wilcoxon signed-rank test; "
        "p-values are adjusted by Benjamini-Hochberg correction across all network-beta tests."
    )

    print("=" * 80)
    print("Fixed-beta SIR final infection scale")
    print("=" * 80)
    print(f"Networks: {networks}")
    print(f"Methods: {methods}")
    print(f"Betas: {betas}")
    print(f"gamma={args.gamma}, seed ratio={args.seed_ratio:.2%}, seed count={args.seed_count}, blocks={args.blocks}, repeats={args.repeats}")
    print(f"Method implementation source: {METHOD_IMPLEMENTATION_SOURCE}")
    print(f"Use precomputed rankings: {args.use_precomputed}")
    print("=" * 80)

    set_seed(args.master_seed)

    final_records: List[Dict] = []
    method_records: List[Dict] = []
    raw_records: List[Dict] = []
    ranking_records: List[Dict] = []

    prefix = make_output_prefix(args)
    checkpoint_name = f"{prefix}_Checkpoint.xlsx"
    final_name = f"{prefix}_Final_Statistical_Test_Summary.xlsx"

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
            if args.seed_count and args.seed_count > 0:
                k = min(n, max(1, int(args.seed_count)))
            else:
                k = max(1, int(n * args.seed_ratio))
            budget_desc = seed_budget_description(args, k)
            net_label = format_network_name(net)
            print(f"  Nodes={n}, Edges={e}, k={k} ({budget_desc})")

            rankings = compute_rankings_once(methods, graph, net, args.use_precomputed)
            seed_sets = {}
            for m in methods:
                seeds = select_top_k(rankings[m], k)
                seed_sets[m] = seeds
                ranking_records.append({
                    "Network": net_label,
                    "Method": m,
                    "MethodLabel": method_label(m, args.proposed_method, args.proposed_label),
                    "Seed_Budget": budget_desc,
                    "k": k,
                    "TopK_Seeds": ",".join(map(str, seeds)),
                    "UsedPrecomputed": args.use_precomputed,
                })

            for beta_idx, beta in enumerate(betas):
                print(f"  [Beta] {beta:.3f}")
                method_blocks = {}

                for m in tqdm(methods, desc=f"    beta={beta:.3f}", leave=False):
                    blocks = evaluate_blocks(
                        graph, seed_sets[m], beta=beta, gamma=args.gamma,
                        network_name=net,
                        method_setting_key=f"k{k}",
                        blocks=args.blocks, repeats=args.repeats,
                        master_seed=args.master_seed, max_steps=args.max_steps,
                    )
                    method_blocks[m] = blocks
                    mean_val, std_val, ci_val = mean_ci95(blocks)
                    method_records.append({
                        "Network": net_label,
                        "N": n,
                        "E": e,
                        "Beta": beta,
                        "Gamma": args.gamma,
                        "Seed_Ratio": args.seed_ratio,
                        "Seed_Budget": budget_desc,
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
                            "Beta": beta,
                            "Gamma": args.gamma,
                            "Seed_Ratio": args.seed_ratio,
                            "Seed_Budget": budget_desc,
                            "k": k,
                            "Method": m,
                            "Block": b_idx,
                            "Final_Infection(%)": val,
                        })

                if args.proposed_method not in method_blocks:
                    continue
                hosh_blocks = method_blocks[args.proposed_method]
                hosh_mean, hosh_std, hosh_ci = mean_ci95(hosh_blocks)

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
                diff_blocks = hosh_blocks - best_blocks
                delta_mean = float(np.mean(diff_blocks))
                diff_std = float(np.std(diff_blocks, ddof=1)) if len(diff_blocks) > 1 else 0.0
                dz = delta_mean / diff_std if diff_std > 0 else np.nan
                win_rate = float(np.mean(diff_blocks > 0) * 100.0)
                improvement = (hosh_mean - best_mean) / best_mean * 100.0 if best_mean > 0 else np.nan

                try:
                    _, p_val = wilcoxon(hosh_blocks, best_blocks, alternative="two-sided")
                    p_val = max(float(p_val), 1e-20)
                except ValueError:
                    p_val = 1.0

                final_records.append({
                    "Network": net_label,
                    "N": n,
                    "E": e,
                    "Beta": beta,
                    "Gamma": args.gamma,
                    "Seed_Ratio": args.seed_ratio,
                    "Seed_Budget": budget_desc,
                    "k": k,
                    "Best_Baseline_Final": best_name,
                    "Best_Baseline_Label": method_label(best_name, args.proposed_method, args.proposed_label),
                    "MSH_Final(%)": hosh_mean,
                    "MSH_95%_CI": hosh_ci,
                    "MSH_Final_Mean±CI": fmt_mean_ci(hosh_mean, hosh_ci),
                    "Best_Final(%)": best_mean,
                    "Best_95%_CI": best_ci,
                    "Best_Final_Mean±CI": fmt_mean_ci(best_mean, best_ci),
                    "Delta_Final(%)": hosh_mean - best_mean,
                    "Mean_Paired_Difference(%)": delta_mean,
                    "Improvement(%)": improvement,
                    "Paired_Effect_dz": dz,
                    "Win_Rate_Blocks(%)": win_rate,
                    "P_Value_Final_vs_Best": p_val,
                })
                print(f"    Best={best_name}; MSH={hosh_mean:.2f}±{hosh_ci:.2f}, Best={best_mean:.2f}±{best_ci:.2f}, improvement={improvement:.2f}%, p={p_val:.2e}")

                # Save checkpoint after each beta setting.
                checkpoint_df = save_outputs(args.output_dir, final_records, method_records, raw_records, ranking_records,
                                             args, notes, checkpoint_name)
                plot_eta_heatmap(checkpoint_df, args.output_dir, args)
                plot_summary_table(checkpoint_df, args.output_dir, args)
                gc.collect()

        except Exception as exc:
            print(f"  [Error] Failed on network {net}: {exc}")
            traceback.print_exc()
            continue

    final_df = save_outputs(args.output_dir, final_records, method_records, raw_records, ranking_records,
                            args, notes, final_name)
    plot_eta_heatmap(final_df, args.output_dir, args)
    plot_summary_table(final_df, args.output_dir, args)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Final workbook: {os.path.join(args.output_dir, final_name)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
