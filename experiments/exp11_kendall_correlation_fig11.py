"""
Reviewer experiment: all-node Kendall rank-correlation diagnostics for MSH.

Purpose
-------
This experiment addresses the reviewer's request for a complete all-node
rank-correlation analysis between MSH and related structural information.

For each of the nine empirical networks, Kendall's tau-b is computed between
MSH and:
    1) Degree centrality (DC)
    2) Maximal-clique participation count |M(v)|
    3) Standard Structural Holes (SH)
    4) Improved Structural Holes (ISH)
    5) SNIM (clique-based baseline)

Why Kendall tau-b?
------------------
Several structural measures contain tied node scores. Kendall's tau-b explicitly
accounts for ties and is therefore appropriate for comparing node orderings.

Important implementation choices
--------------------------------
* The repository internally names the proposed method "HOSH"; manuscript output
  is always labelled "MSH".
* The script uses RAW METHOD SCORES, not ID-based tie-broken final rankings.
* No node-ID tie breaking is used in the correlation calculation.
* scipy.stats.kendalltau(..., variant="b") handles tied values.
* MSH/SH/ISH are assumed to be stored by hosh_methods.py in the same
  influence-oriented direction used by the repository. If your local repository
  uses the opposite direction for MSH, set MSH_DIRECTION = -1 below.
* Clique participation is exactly the number of maximal cliques containing a
  node, |M(v)|, including size-2 maximal cliques when they are maximal cliques.
* This is a deterministic structural diagnostic. No Monte Carlo confidence
  interval, Wilcoxon test, or significance star is added.

Interpretation
--------------
tau_b > 0 : overall concordant ranking tendency
tau_b ~ 0 : weak rank association
tau_b < 0 : overall inverse ranking tendency

A low or negative correlation means that MSH produces a different ordering.
It does NOT, by itself, demonstrate better spreading performance.

Outputs
-------
results/comment2_kendall_rank_correlations/
    Table_C2_Kendall_by_network.csv
    Table_C2_Kendall_summary.csv
    Table_C2_AllNodeScoreDiagnostics.csv
    Table_C2_Kendall.xlsx
    Fig_C2_MSH_Kendall_heatmap.pdf
    Fig_C2_MSH_Kendall_heatmap.png
    EXPERIMENT_REPORT.md

Run
---
Formal run:
    python exp_comment2_structural_rank_correlations_kendall.py

Selected networks:
    python exp_comment2_structural_rank_correlations_kendall.py --networks lesmis adjnoun jazz

Force recomputation of required method scores:
    python exp_comment2_structural_rank_correlations_kendall.py --force
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import kendalltau

from network_loader import download_and_load_graph, get_network_list
from hosh_methods import build_clique_data, get_node_scores


# =============================================================================
# 0. Configuration
# =============================================================================

OUTPUT_DIR = Path("results/comment2_kendall_rank_correlations")
RANKING_DIR = Path("results/node_rankings")
CLIQUE_CACHE_DIR = Path("results/clique_cache")

INTERNAL_MSH = "HOSH"
METHODS_NEEDED = [INTERNAL_MSH, "DC", "SH", "ISH", "SNIM"]

CORRELATES = [
    "Degree",
    "Clique participation",
    "SH",
    "ISH",
    "SNIM",
]

NETWORK_DISPLAY = {
    "lesmis": "Lesmis",
    "adjnoun": "Adjnoun",
    "jazz": "Jazz",
    "usair": "Usair",
    "infect": "Infect",
    "email": "Email",
    "polblogs": "Polblogs",
    "hamster": "Hamster",
    "power": "Power",
}

# IMPORTANT:
# Keep +1 if hosh_methods.py already returns MSH/HOSH in the manuscript-facing
# influence direction (larger value = higher importance).
# Change to -1 ONLY if your local HOSH raw score uses the opposite direction.
MSH_DIRECTION = +1.0

# Numerical rounding used ONLY to diagnose ties, not to compute Kendall tau-b.
TIE_DIAGNOSTIC_DECIMALS = 12


# =============================================================================
# 1. Plot style
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
})


# =============================================================================
# 2. Cache helpers
# =============================================================================

def safe_pickle_load(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def safe_pickle_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_or_build_clique_data(
    network_name: str,
    g: nx.Graph,
    force: bool = False,
):
    """Reuse clique cache when available; otherwise build it once."""
    cache_file = CLIQUE_CACHE_DIR / f"{network_name}_cliques.pkl"

    if cache_file.exists() and not force:
        cached = safe_pickle_load(cache_file)
        if (
            isinstance(cached, dict)
            and "cliques" in cached
            and "node_cliques_map" in cached
        ):
            return cached

    clique_data = build_clique_data(g)
    safe_pickle_save(clique_data, cache_file)
    return clique_data


def load_or_compute_scores(
    network_name: str,
    g: nx.Graph,
    clique_data: Mapping,
    force: bool = False,
) -> Dict[str, Dict]:
    """
    Load existing node-score cache and compute only required missing methods.

    The diagnostic deliberately uses score dictionaries rather than a final
    ID-tie-broken ranking list.
    """
    ranking_file = RANKING_DIR / f"{network_name}_rankings.pkl"
    rankings = {} if force else (safe_pickle_load(ranking_file) or {})

    changed = False

    for method in METHODS_NEEDED:
        valid = (
            method in rankings
            and isinstance(rankings[method], dict)
            and len(rankings[method]) == g.number_of_nodes()
        )

        if valid and not force:
            continue

        rankings[method] = get_node_scores(
            method,
            g,
            clique_data=clique_data,
        )
        changed = True

    if changed or not ranking_file.exists():
        safe_pickle_save(rankings, ranking_file)

    return {method: rankings[method] for method in METHODS_NEEDED}


# =============================================================================
# 3. Score construction and direction checks
# =============================================================================

def clique_participation_scores(
    g: nx.Graph,
    clique_data: Mapping,
) -> Dict:
    """
    CP_i = |M(v_i)|:
    number of maximal cliques containing node i.
    """
    node_cliques_map = clique_data["node_cliques_map"]
    return {
        node: float(len(node_cliques_map.get(node, [])))
        for node in g.nodes()
    }


def apply_msh_direction(scores: Mapping) -> Dict:
    """
    Convert MSH/HOSH to the manuscript-facing influence direction.

    With the current repository implementation this should normally remain +1.
    """
    return {
        node: MSH_DIRECTION * float(value)
        for node, value in scores.items()
    }


def aligned_values(
    nodes: Iterable,
    x: Mapping,
    y: Mapping,
) -> Tuple[np.ndarray, np.ndarray]:
    xv = np.asarray([float(x[n]) for n in nodes], dtype=float)
    yv = np.asarray([float(y[n]) for n in nodes], dtype=float)

    valid = np.isfinite(xv) & np.isfinite(yv)
    return xv[valid], yv[valid]


# =============================================================================
# 4. Kendall tau-b
# =============================================================================

def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """
    Kendall's tau-b with explicit tie correction.

    If one score vector is constant, the coefficient is undefined and NaN is
    returned. No arbitrary node-ID ordering is introduced.
    """
    if x.size < 2 or y.size < 2:
        return np.nan

    if np.unique(x).size <= 1 or np.unique(y).size <= 1:
        return np.nan

    result = kendalltau(
        x,
        y,
        variant="b",
        nan_policy="omit",
    )
    return float(result.statistic)


def tie_statistics(values: np.ndarray) -> Tuple[int, float]:
    """
    Descriptive tie diagnostics only.

    Returns:
        number of unique scores,
        tie fraction = 1 - unique_scores / N
    """
    if values.size == 0:
        return 0, np.nan

    rounded = np.round(values, TIE_DIAGNOSTIC_DECIMALS)
    n_unique = int(np.unique(rounded).size)
    tie_fraction = 1.0 - n_unique / len(values)
    return n_unique, float(tie_fraction)


# =============================================================================
# 5. Per-network analysis
# =============================================================================

def analyze_network(
    network_name: str,
    force: bool = False,
) -> Tuple[dict, List[dict]]:
    g = download_and_load_graph(network_name, verbose=False)
    if g is None:
        raise RuntimeError(f"Could not load network: {network_name}")

    clique_data = load_or_build_clique_data(
        network_name,
        g,
        force=False,
    )

    scores = load_or_compute_scores(
        network_name,
        g,
        clique_data,
        force=force,
    )

    msh = apply_msh_direction(scores[INTERNAL_MSH])

    comparators = {
        "Degree": scores["DC"],
        "Clique participation": clique_participation_scores(g, clique_data),
        "SH": scores["SH"],
        "ISH": scores["ISH"],
        "SNIM": scores["SNIM"],
    }

    # Node order is used only to align score vectors.
    # It is NOT used to break ranking ties.
    nodes = list(g.nodes())

    row = {
        "Network": NETWORK_DISPLAY.get(network_name, network_name),
        "N": g.number_of_nodes(),
        "E": g.number_of_edges(),
        "Maximal_cliques": len(clique_data["cliques"]),
    }

    diagnostics: List[dict] = []

    msh_values = np.asarray([float(msh[n]) for n in nodes], dtype=float)
    msh_unique, msh_tie_fraction = tie_statistics(msh_values)

    row["MSH_unique_scores"] = msh_unique
    row["MSH_tie_fraction"] = msh_tie_fraction

    for label in CORRELATES:
        x, y = aligned_values(nodes, msh, comparators[label])
        tau = kendall_tau_b(x, y)

        comparator_unique, comparator_tie_fraction = tie_statistics(y)

        row[label] = tau

        diagnostics.append({
            "Network": NETWORK_DISPLAY.get(network_name, network_name),
            "Comparator": label,
            "N_used": int(len(x)),
            "Kendall_tau_b": tau,
            "MSH_unique_scores": msh_unique,
            "Comparator_unique_scores": comparator_unique,
            "MSH_tie_fraction": msh_tie_fraction,
            "Comparator_tie_fraction": comparator_tie_fraction,
        })

    return row, diagnostics


# =============================================================================
# 6. Across-network descriptive summary
# =============================================================================

def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in CORRELATES:
        values = (
            pd.to_numeric(df[col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            rows.append({
                "Comparator": col,
                "Mean_tau_b": np.nan,
                "Median_tau_b": np.nan,
                "Min_tau_b": np.nan,
                "Max_tau_b": np.nan,
                "Mean_abs_tau_b": np.nan,
                "Networks_abs_tau_b_ge_0.5": 0,
            })
            continue

        rows.append({
            "Comparator": col,
            "Mean_tau_b": float(np.mean(values)),
            "Median_tau_b": float(np.median(values)),
            "Min_tau_b": float(np.min(values)),
            "Max_tau_b": float(np.max(values)),
            "Mean_abs_tau_b": float(np.mean(np.abs(values))),
            "Networks_abs_tau_b_ge_0.5": int(
                np.sum(np.abs(values) >= 0.5)
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 7. Heatmap
# =============================================================================

def plot_heatmap(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    plot_df = df.set_index("Network")[CORRELATES].astype(float)
    data = plot_df.to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(CORRELATES)))
    ax.set_xticklabels(
        CORRELATES,
        rotation=25,
        ha="right",
        fontsize=10,
    )

    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=10)

    # Keep the figure concise; the caption explains the comparison.
    ax.set_xlabel("")
    ax.set_ylabel("Network", fontsize=11)

    # Annotate exact tau-b values.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "NA" if np.isnan(val) else f"{val:.2f}"

            text_color = (
                "white"
                if (not np.isnan(val) and abs(val) >= 0.50)
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.03,
    )
    cbar.set_label(
        r"Kendall $\tau_b$",
        fontsize=11,
    )
    cbar.ax.tick_params(labelsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. Report
# =============================================================================

def qualitative_label(tau: float) -> str:
    """
    Only for descriptive report text.
    Avoid rigid threshold claims in the manuscript.
    """
    if not np.isfinite(tau):
        return "undefined"

    a = abs(tau)

    if a < 0.10:
        return "very weak"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "relatively strong"
    return "very strong"


def write_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    lines: List[str] = []

    lines.append(
        "# Reviewer experiment: MSH all-node Kendall rank-correlation analysis\n"
    )

    lines.append("## 1. Objective\n")
    lines.append(
        "Assess whether MSH largely reproduces degree, maximal-clique participation, "
        "standard structural-hole information, ISH, or the clique-based SNIM ordering, "
        "or instead produces a network-dependent reorganization of these structural signals.\n"
    )

    lines.append("## 2. Method\n")
    lines.append(
        "For each of the nine processed empirical networks, Kendall's tau-b was computed "
        "across all nodes between MSH and five structural comparators: Degree, maximal-clique "
        "participation count |M(v)|, SH, ISH, and SNIM. Kendall's tau-b was selected because "
        "several node measures contain tied scores and tau-b explicitly adjusts for ties. "
        "No node-ID tie breaking was used in the correlation calculation.\n"
    )

    lines.append("## 3. Per-network Kendall tau-b\n")
    display_cols = ["Network"] + CORRELATES
    lines.append(
        df[display_cols]
        .round(4)
        .to_markdown(index=False)
    )
    lines.append("\n")

    lines.append("## 4. Across-network descriptive summary\n")
    lines.append(
        summary
        .round(4)
        .to_markdown(index=False)
    )
    lines.append("\n")

    lines.append("## 5. Tie diagnostics\n")
    lines.append(
        diagnostics
        .round(4)
        .to_markdown(index=False)
    )
    lines.append("\n")

    lines.append("## 6. Interpretation rule\n")
    lines.append(
        "Kendall tau-b is used here as a diagnostic of ranking agreement rather than "
        "a performance metric. Positive values indicate an overall concordant ordering "
        "tendency, values close to zero indicate weak rank association, and negative values "
        "indicate an inverse ordering tendency. A lower or negative coefficient does not "
        "by itself demonstrate superior influence identification; spreading performance is "
        "evaluated separately by the multi-seed SIR experiments.\n"
    )

    lines.append("## 7. Automatic descriptive reading\n")
    for _, row in summary.iterrows():
        median_tau = (
            float(row["Median_tau_b"])
            if pd.notna(row["Median_tau_b"])
            else np.nan
        )
        lines.append(
            f"- **{row['Comparator']}**: median tau-b = "
            f"{median_tau:.3f} "
            f"({qualitative_label(median_tau)} rank association in the "
            f"nine-network descriptive summary)."
        )

    lines.append("\n")

    lines.append("## 8. Manuscript claim boundary\n")
    lines.append(
        "Use the results to characterize whether MSH is a simple ranking proxy for a single "
        "existing descriptor/baseline or whether it reorganizes structural information in a "
        "network-dependent manner. Do not interpret low correlation as evidence of higher "
        "ranking accuracy. The ablation experiment evaluates the contribution of individual "
        "MSH components, while the multi-seed SIR experiments evaluate spreading performance.\n"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 9. Main workflow
# =============================================================================

def main(
    networks: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RANKING_DIR.mkdir(parents=True, exist_ok=True)
    CLIQUE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if networks is None:
        networks = get_network_list()

    print("=" * 82)
    print("MSH all-node Kendall rank-correlation diagnostic")
    print("Statistic: Kendall's tau-b")
    print("Comparators: Degree, clique participation, SH, ISH, SNIM")
    print("Tie handling: tau-b; NO node-ID tie breaking")
    print(f"MSH direction multiplier: {MSH_DIRECTION:+.0f}")
    print("=" * 82)

    rows: List[dict] = []
    diagnostics: List[dict] = []
    failures: List[Tuple[str, str]] = []

    for idx, network_name in enumerate(networks, start=1):
        print(
            f"[{idx}/{len(networks)}] {network_name} ... ",
            end="",
            flush=True,
        )

        try:
            row, diag = analyze_network(
                network_name,
                force=force,
            )

            rows.append(row)
            diagnostics.extend(diag)

            tau_text = ", ".join(
                f"{col}={row[col]:.3f}"
                if np.isfinite(row[col])
                else f"{col}=NA"
                for col in CORRELATES
            )
            print(tau_text)

        except Exception as exc:
            failures.append((network_name, str(exc)))
            print(f"FAILED: {exc}")

    if not rows:
        raise RuntimeError("No network was successfully analyzed.")

    df = pd.DataFrame(rows)

    # Preserve the manuscript's network order.
    display_order = [
        NETWORK_DISPLAY.get(name, name)
        for name in networks
    ]

    order_map = {
        name: idx
        for idx, name in enumerate(display_order)
    }

    df["_order"] = df["Network"].map(order_map)
    df = (
        df.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    diagnostics_df = pd.DataFrame(diagnostics)
    summary_df = correlation_summary(df)

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    table_csv = OUTPUT_DIR / "Table_C2_Kendall_by_network.csv"
    summary_csv = OUTPUT_DIR / "Table_C2_Kendall_summary.csv"
    diagnostics_csv = OUTPUT_DIR / "Table_C2_AllNodeScoreDiagnostics.csv"

    excel_file = OUTPUT_DIR / "Table_C2_Kendall.xlsx"

    fig_pdf = OUTPUT_DIR / "Fig_C2_MSH_Kendall_heatmap.pdf"
    fig_png = OUTPUT_DIR / "Fig_C2_MSH_Kendall_heatmap.png"

    report_file = OUTPUT_DIR / "EXPERIMENT_REPORT.md"

    # -------------------------------------------------------------------------
    # Save numerical results
    # -------------------------------------------------------------------------
    df.to_csv(
        table_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    diagnostics_df.to_csv(
        diagnostics_csv,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    manuscript_table = df[
        ["Network"] + CORRELATES
    ].copy()

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:
        manuscript_table.to_excel(
            writer,
            sheet_name="Manuscript",
            index=False,
        )
        df.to_excel(
            writer,
            sheet_name="Full_results",
            index=False,
        )
        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        diagnostics_df.to_excel(
            writer,
            sheet_name="Tie_diagnostics",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Figure and report
    # -------------------------------------------------------------------------
    plot_heatmap(
        df,
        fig_pdf,
        fig_png,
    )

    write_report(
        df,
        summary_df,
        diagnostics_df,
        report_file,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\nSaved:")
    for path in [
        table_csv,
        summary_csv,
        diagnostics_csv,
        excel_file,
        fig_pdf,
        fig_png,
        report_file,
    ]:
        print(f"  - {path}")

    print("\nManuscript-facing Kendall tau-b table:")
    print(
        manuscript_table
        .round(4)
        .to_string(index=False)
    )

    print("\nAcross-network descriptive summary:")
    print(
        summary_df
        .round(4)
        .to_string(index=False)
    )

    if failures:
        print("\nNetworks that failed:")
        for name, error in failures:
            print(f"  - {name}: {error}")

        print(
            "\nRe-run only the failed network(s) after fixing local data availability."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "All-node Kendall tau-b correlations between MSH "
            "and structural comparators."
        )
    )

    parser.add_argument(
        "--networks",
        nargs="+",
        default=None,
        help=(
            "Network names to analyze. "
            "Default: all networks returned by network_loader.py."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute MSH/DC/SH/ISH/SNIM score dictionaries "
            "even if local cache files exist."
        ),
    )

    args = parser.parse_args()

    main(
        networks=args.networks,
        force=args.force,
    )
