#!/usr/bin/env python3
"""
Revised synthetic runtime experiment: fixed average degree + increasing network size + baseline comparison.

Purpose
-------
This script compares the ranking runtime of MSH/HOSH and baseline methods on
controlled synthetic networks. The network size N increases while the target
average degree is kept approximately fixed. Graph generation time is excluded;
all methods are timed on the same preprocessed graph instance.

Default protocol
----------------
- Models: ER, BA, WS
- N values: 1000, 2000, 5000, 10000, 20000
- Target average degree: 6
- Independent graph instances per condition: 5
- Timing repeats per instance: 1 by default, adjustable by --repeat-times
- Methods: HOSH/MSH, VoteRank, SNIM, CHBC, ISH, DC, BC, CC, K-Shell, SH, CI, SNC

Important implementation notes
------------------------------
- Existing runtime summary files are reused for plotting by default when available; use --force-rerun to recompute timings.
- No precomputed rankings or clique caches are read when timings are recomputed.
- Each timing repeat recomputes the specified method's ranking scores from scratch.
- HOSH and SNIM therefore include their own maximal-clique enumeration cost.
- CHBC includes its own Louvain partition computation through get_node_scores().
- Python's randomized built-in hash() is not used for seed derivation.

Outputs
-------
results/exp_runtime_synthetic_scale_baseline_comparison/
    Runtime_Synthetic_Scale_Baseline_Comparison.xlsx
    Runtime_Synthetic_Scale_Baseline_Comparison_Checkpoint.xlsx
    Runtime_Synthetic_Scale_Baseline_Comparison_<model>.pdf/png
    Runtime_Synthetic_Scale_Baseline_Legend.pdf/png
    Runtime_Synthetic_Scale_Baseline_Method_Summary.csv
    Runtime_Synthetic_Scale_Baseline_Raw.csv
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
import shutil
import sys
import time
import traceback
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, FixedLocator, NullFormatter
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import t

# Prefer the optimized exact MSH/HOSH implementation.
# The loader supports two deployment styles:
#   1) keep hosh_methods_fast_final.py in the project folder; or
#   2) replace hosh_methods.py with the optimized implementation.
# By default the experiment requires an optimized implementation; use
# --allow-unoptimized-fallback only for debugging.
def _load_method_implementation():
    import importlib

    last_error = None
    for module_name in ("hosh_methods_fast_final", "hosh_methods_optimized", "hosh_methods"):
        try:
            module = importlib.import_module(module_name)
            get_scores = getattr(module, "get_node_scores")

            # Optimized files expose enumerate_maximal_cliques() and avoid repeated
            # set.intersection() in the HOSH overlap-redundancy stage. If the
            # optimized file has been copied to hosh_methods.py, this flag remains true.
            optimized = (
                module_name in {"hosh_methods_fast_final", "hosh_methods_optimized"}
                or hasattr(module, "enumerate_maximal_cliques")
            )
            return get_scores, module_name, optimized
        except Exception as exc:  # pragma: no cover - import diagnostics
            last_error = exc
            continue

    raise ImportError(f"Could not import any HOSH/MSH method implementation. Last error: {last_error}")


get_node_scores, METHOD_IMPLEMENTATION_SOURCE, OPTIMIZED_MSH_ACTIVE = _load_method_implementation()


# =============================================================================
# 0. Plot configuration consistent with the other revised experiments
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

MARKERS = {
    "MSH": "o",
    "HOSH": "o",
    "VoteRank": "o",
    "SNIM": "p",
    "CHBC": "*",
    "ISH": "s",
    "DC": "^",
    "BC": "D",
    "CC": "X",
    "K-Shell": "P",
    "SH": "v",
    "CI": "h",
    "SNC": "H",
}


# =============================================================================
# 1. Reproducibility and utility functions
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(*items: object, modulus: int = 1_000_000_000) -> int:
    """Deterministic integer hash independent of Python hash randomization."""
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % modulus


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def summarize_numeric(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "ci95": np.nan, "min": np.nan, "max": np.nan}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(t.ppf(0.975, df=arr.size - 1) * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": std,
        "ci95": float(ci95),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ci95(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(t.ppf(0.975, df=len(arr) - 1) * float(np.std(arr, ddof=1)) / math.sqrt(len(arr)))


def method_label(method: str, proposed_method: str, proposed_label: str) -> str:
    return proposed_label if method == proposed_method else method


def remove_previous_outputs(output_dir: str, filenames: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in filenames:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def style_axes(ax) -> None:
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#000000")
    ax.tick_params(direction="out", which="major", length=3.0, width=0.7)


def format_scientific_tick(x, pos=None) -> str:
    """Format network-size ticks as compact scientific notation."""
    try:
        x = float(x)
    except Exception:
        return ""
    if not np.isfinite(x) or x <= 0:
        return ""
    exponent = int(np.floor(np.log10(x)))
    coeff = x / (10 ** exponent)
    if np.isclose(coeff, 1.0):
        return rf"$10^{{{exponent}}}$"
    if np.isclose(coeff, round(coeff)):
        coeff_text = str(int(round(coeff)))
    else:
        coeff_text = f"{coeff:.1f}".rstrip("0").rstrip(".")
    return rf"${coeff_text}\times10^{{{exponent}}}$"


def format_model_title(model: str) -> str:
    """Keep synthetic model names in the same title position as network names in SIR figures."""
    return str(model).upper()


# =============================================================================
# 2. Synthetic graph generation and preprocessing
# =============================================================================
def make_er(n: int, avg_degree: float, seed: int) -> nx.Graph:
    p = min(max(avg_degree / max(n - 1, 1), 0.0), 1.0)
    return nx.erdos_renyi_graph(n, p, seed=seed)


def make_ba(n: int, avg_degree: float, seed: int) -> nx.Graph:
    m = max(1, int(round(avg_degree / 2.0)))
    m = min(m, max(1, n - 1))
    return nx.barabasi_albert_graph(n, m, seed=seed)


def make_ws(n: int, avg_degree: float, rewiring_p: float, seed: int) -> nx.Graph:
    k = max(2, int(round(avg_degree)))
    if k % 2 == 1:
        k += 1
    k = min(k, max(2, n - 1))
    if k % 2 == 1:
        k -= 1
    k = max(2, k)
    return nx.watts_strogatz_graph(n, k, rewiring_p, seed=seed)


def generate_graph(model: str, n: int, avg_degree: float, ws_rewiring: float, seed: int) -> nx.Graph:
    model = model.upper()
    if model == "ER":
        return make_er(n, avg_degree, seed)
    if model == "BA":
        return make_ba(n, avg_degree, seed)
    if model == "WS":
        return make_ws(n, avg_degree, ws_rewiring, seed)
    raise ValueError(f"Unknown synthetic model: {model}")


def preprocess_graph(g_raw: nx.Graph) -> Tuple[nx.Graph, Dict[str, object]]:
    """Apply the same simple-graph/LCC/relabel protocol used by revised experiments."""
    t0 = time.perf_counter()
    n0 = g_raw.number_of_nodes()
    e0 = g_raw.number_of_edges()

    g = nx.Graph(g_raw)
    g.remove_edges_from(nx.selfloop_edges(g))

    if g.number_of_nodes() > 0 and not nx.is_connected(g):
        lcc_nodes = max(nx.connected_components(g), key=len)
        g = g.subgraph(lcc_nodes).copy()

    lcc_retention = g.number_of_nodes() / max(n0, 1)
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    elapsed = time.perf_counter() - t0

    return g, {
        "OriginalN": n0,
        "OriginalE": e0,
        "LCCRetention": lcc_retention,
        "PreprocessTime_s": elapsed,
    }


def graph_degeneracy(g: nx.Graph) -> int:
    try:
        core = nx.core_number(g)
        return int(max(core.values())) if core else 0
    except Exception:
        return 0


def graph_structural_stats(g: nx.Graph) -> Dict[str, object]:
    n = g.number_of_nodes()
    e = g.number_of_edges()
    degrees = [d for _, d in g.degree()]
    return {
        "N": n,
        "E": e,
        "AvgDegree_Actual": float(np.mean(degrees)) if degrees else 0.0,
        "Density": float(nx.density(g)) if n > 1 else 0.0,
        "AvgClustering": float(nx.average_clustering(g)) if n > 0 else np.nan,
        "Transitivity": float(nx.transitivity(g)) if n > 0 else np.nan,
        "Degeneracy": graph_degeneracy(g),
    }


# =============================================================================
# 3. Method timing
# =============================================================================
def run_method_once(method: str, g: nx.Graph, run_seed: int) -> Tuple[Optional[float], Optional[int], str]:
    """Run one ranking method once and return elapsed time, score count, and error message."""
    set_seed(run_seed)
    gc.collect()
    try:
        t0 = time.perf_counter()
        scores = get_node_scores(method, g)
        elapsed = time.perf_counter() - t0
        score_count = len(scores) if hasattr(scores, "__len__") else None
        return elapsed, score_count, ""
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


def time_method(method: str, g: nx.Graph, model: str, n_target: int, instance: int, repeat_times: int,
                master_seed: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    raw_rows: List[Dict[str, object]] = []
    for rep in range(repeat_times):
        run_seed = master_seed + stable_int_hash("method", model, n_target, instance, method, rep, modulus=1_000_000_000)
        elapsed, score_count, err = run_method_once(method, g, run_seed)
        raw_rows.append({
            "Method": method,
            "Repeat": rep,
            "Runtime_s": elapsed,
            "ScoreCount": score_count,
            "RunSeed": run_seed,
            "RunError": err,
        })

    stats = summarize_numeric(row["Runtime_s"] for row in raw_rows)
    summary = {
        "Method": method,
        "SuccessRuns": stats["n"],
        "Runtime_Mean_s": stats["mean"],
        "Runtime_Median_s": stats["median"],
        "Runtime_Std_s": stats["std"],
        "Runtime_CI95_s": stats["ci95"],
        "Runtime_Min_s": stats["min"],
        "Runtime_Max_s": stats["max"],
        "FailedRuns": sum(1 for row in raw_rows if row.get("RunError")),
    }
    return raw_rows, summary


def run_one_instance(model: str, n_target: int, avg_degree: float, ws_rewiring: float, instance: int,
                     methods: List[str], args) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    graph_seed = args.master_seed + stable_int_hash("graph", model, n_target, avg_degree, ws_rewiring, instance, modulus=1_000_000_000)
    set_seed(graph_seed)
    g_raw = generate_graph(model, n_target, avg_degree, ws_rewiring, graph_seed)
    g, prep = preprocess_graph(g_raw)
    g_stats = graph_structural_stats(g)

    instance_meta = {
        "NetworkModel": model,
        "N_Target": n_target,
        "TargetAvgDegree": avg_degree,
        "WS_RewiringP": ws_rewiring,
        "Instance": instance,
        "GraphSeed": graph_seed,
        **prep,
        **g_stats,
    }

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for method in methods:
        method_raw, method_summary = time_method(
            method, g, model=model, n_target=n_target, instance=instance,
            repeat_times=args.repeat_times, master_seed=args.master_seed,
        )

        for row in method_raw:
            raw_rows.append({**instance_meta, **row})
        summary_rows.append({**instance_meta, **method_summary})

    return raw_rows, summary_rows, instance_meta


# =============================================================================
# 4. Summary, export, and plotting
# =============================================================================
def collect_environment_metadata(args, notes: str) -> pd.DataFrame:
    rows = [
        ("Generated at", _dt.datetime.now().isoformat(timespec="seconds")),
        ("Protocol", notes),
        ("Plotting policy", "Existing runtime summary files are reused for plotting by default when available; use --force-rerun to recompute timings."),
        ("Fresh-run policy", "When timings are recomputed, the script does not read precomputed rankings or clique caches."),
        ("Runtime scope", "Graph generation is excluded. Method runtime is measured on the preprocessed graph. HOSH/SNIM include their internal maximal-clique enumeration; CHBC includes its own partition computation."),
        ("Synthetic models", args.models),
        ("N values", args.sizes),
        ("Target average degree", str(args.avg_degree)),
        ("WS rewiring probability", str(args.ws_rewiring)),
        ("Instances per condition", str(args.instances)),
        ("Timing repeats per instance", str(args.repeat_times)),
        ("Master seed", str(args.master_seed)),
        ("Proposed method in code", args.proposed_method),
        ("Proposed label in figures/tables", args.proposed_label),
        ("Method implementation source", METHOD_IMPLEMENTATION_SOURCE),
        ("Optimized MSH/HOSH implementation active", str(OPTIMIZED_MSH_ACTIVE)),
        ("Python version", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("Processor", platform.processor()),
        ("NetworkX version", getattr(nx, "__version__", "unknown")),
        ("NumPy version", getattr(np, "__version__", "unknown")),
        ("Pandas version", getattr(pd, "__version__", "unknown")),
        ("Matplotlib version", getattr(matplotlib, "__version__", "unknown")),
    ]
    try:
        import igraph as ig  # type: ignore
        rows.append(("igraph version", getattr(ig, "__version__", "available")))
    except Exception:
        rows.append(("igraph version", "not available"))
    try:
        import community.community_louvain as community_louvain  # type: ignore
        rows.append(("python-louvain version", getattr(community_louvain, "__version__", "available")))
    except Exception:
        rows.append(("python-louvain version", "not available"))
    try:
        import psutil  # type: ignore
        rows.extend([
            ("Physical cores", str(psutil.cpu_count(logical=False))),
            ("Logical cores", str(psutil.cpu_count(logical=True))),
            ("RAM GB", f"{psutil.virtual_memory().total / (1024 ** 3):.2f}"),
        ])
    except Exception:
        rows.append(("psutil hardware metadata", "not available"))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def build_condition_summary(instance_summary_df: pd.DataFrame, methods: List[str], args) -> pd.DataFrame:
    if instance_summary_df.empty:
        return pd.DataFrame()
    df = instance_summary_df.copy()
    df["MethodLabel"] = df["Method"].map(lambda m: method_label(m, args.proposed_method, args.proposed_label))
    group_cols = ["NetworkModel", "N_Target", "TargetAvgDegree", "Method", "MethodLabel"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            Instances=("Instance", "nunique"),
            SuccessfulInstances=("SuccessRuns", lambda x: int(np.sum(pd.to_numeric(x, errors="coerce") > 0))),
            Runtime_Mean_s=("Runtime_Mean_s", "mean"),
            Runtime_MedianAcrossInstances_s=("Runtime_Median_s", "median"),
            Runtime_CI95_s=("Runtime_Mean_s", ci95),
            FailedRuns=("FailedRuns", "sum"),
            N_Mean=("N", "mean"),
            E_Mean=("E", "mean"),
            AvgDegree_Actual_Mean=("AvgDegree_Actual", "mean"),
            AvgClustering_Mean=("AvgClustering", "mean"),
            Degeneracy_Mean=("Degeneracy", "mean"),
            LCCRetention_Mean=("LCCRetention", "mean"),
            PreprocessTime_Mean_s=("PreprocessTime_s", "mean"),
        )
        .reset_index()
    )
    return summary


def build_graph_summary(graph_rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not graph_rows:
        return pd.DataFrame()
    gdf = pd.DataFrame(graph_rows).drop_duplicates(subset=["NetworkModel", "N_Target", "Instance"])
    summary = (
        gdf.groupby(["NetworkModel", "N_Target", "TargetAvgDegree"], dropna=False)
        .agg(
            Instances=("Instance", "nunique"),
            N_Mean=("N", "mean"),
            N_CI95=("N", ci95),
            E_Mean=("E", "mean"),
            E_CI95=("E", ci95),
            AvgDegree_Actual_Mean=("AvgDegree_Actual", "mean"),
            AvgDegree_Actual_CI95=("AvgDegree_Actual", ci95),
            Density_Mean=("Density", "mean"),
            AvgClustering_Mean=("AvgClustering", "mean"),
            Transitivity_Mean=("Transitivity", "mean"),
            Degeneracy_Mean=("Degeneracy", "mean"),
            LCCRetention_Mean=("LCCRetention", "mean"),
            PreprocessTime_Mean_s=("PreprocessTime_s", "mean"),
        )
        .reset_index()
    )
    return summary


def build_runtime_wide_tables(condition_summary: pd.DataFrame, methods: List[str], args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if condition_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ordered_labels = [method_label(m, args.proposed_method, args.proposed_label) for m in methods]
    index_cols = ["NetworkModel", "N_Target"]

    mean_table = condition_summary.pivot_table(index=index_cols, columns="MethodLabel", values="Runtime_Mean_s", aggfunc="first")
    median_table = condition_summary.pivot_table(index=index_cols, columns="MethodLabel", values="Runtime_MedianAcrossInstances_s", aggfunc="first")
    ci_table = condition_summary.pivot_table(index=index_cols, columns="MethodLabel", values="Runtime_CI95_s", aggfunc="first")

    for table in [mean_table, median_table, ci_table]:
        table.sort_index(inplace=True)
        table.columns.name = None
    ordered_existing = [m for m in ordered_labels if m in mean_table.columns]
    return (
        median_table.reindex(columns=ordered_existing).reset_index(),
        mean_table.reindex(columns=ordered_existing).reset_index(),
        ci_table.reindex(columns=ordered_existing).reset_index(),
    )


def save_workbook(raw_rows: List[Dict[str, object]], instance_summary_rows: List[Dict[str, object]],
                  graph_rows: List[Dict[str, object]], methods: List[str], args, path: str, notes: str) -> None:
    raw_df = pd.DataFrame(raw_rows)
    instance_summary_df = pd.DataFrame(instance_summary_rows)
    condition_summary = build_condition_summary(instance_summary_df, methods, args)
    graph_summary = build_graph_summary(graph_rows)
    median_table, mean_table, ci_table = build_runtime_wide_tables(condition_summary, methods, args)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        median_table.to_excel(writer, sheet_name="Runtime_Median_s", index=False)
        mean_table.to_excel(writer, sheet_name="Runtime_Mean_s", index=False)
        ci_table.to_excel(writer, sheet_name="Runtime_CI95_s", index=False)
        condition_summary.to_excel(writer, sheet_name="Condition_Method_Summary", index=False)
        instance_summary_df.to_excel(writer, sheet_name="Instance_Method_Summary", index=False)
        graph_summary.to_excel(writer, sheet_name="Graph_Summary", index=False)
        raw_df.to_excel(writer, sheet_name="Raw", index=False)
        collect_environment_metadata(args, notes).to_excel(writer, sheet_name="Protocol_Metadata", index=False)
        pd.DataFrame({"Notes": [notes]}).to_excel(writer, sheet_name="Notes", index=False)


def export_csv_outputs(output_dir: str, raw_rows: List[Dict[str, object]], instance_summary_rows: List[Dict[str, object]],
                       graph_rows: List[Dict[str, object]], methods: List[str], args) -> None:
    raw_df = pd.DataFrame(raw_rows)
    instance_summary_df = pd.DataFrame(instance_summary_rows)
    condition_summary = build_condition_summary(instance_summary_df, methods, args)
    graph_summary = build_graph_summary(graph_rows)
    raw_df.to_csv(os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Raw.csv"), index=False)
    instance_summary_df.to_csv(os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Instance_Summary.csv"), index=False)
    condition_summary.to_csv(os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Method_Summary.csv"), index=False)
    graph_summary.to_csv(os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Graph_Summary.csv"), index=False)


def load_existing_condition_summary(output_dir: str, methods: List[str], args) -> Optional[pd.DataFrame]:
    """Load previously computed condition summary for plotting without rerunning timings."""
    candidates = [
        os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Method_Summary.csv"),
        os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Comparison.xlsx"),
        os.path.join(output_dir, "Runtime_Synthetic_Scale_Baseline_Comparison_Checkpoint.xlsx"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, sheet_name="Condition_Method_Summary")
            if df.empty:
                continue
            required = {"NetworkModel", "N_Target", "Method"}
            if not required.issubset(df.columns):
                continue
            if "MethodLabel" not in df.columns:
                df["MethodLabel"] = df["Method"].map(lambda m: method_label(m, args.proposed_method, args.proposed_label))
            if "Runtime_CI95_s" not in df.columns:
                df["Runtime_CI95_s"] = 0.0
            if args.plot_stat not in df.columns:
                raise ValueError(f"Existing summary lacks requested plot statistic: {args.plot_stat}")
            print(f"[Input] Existing runtime summary found and loaded: {path}")
            return df
        except Exception as exc:
            print(f"[Warning] Failed to load existing summary from {path}: {exc}")
    return None


def export_standalone_legend(methods: List[str], output_dir: str, args) -> None:
    labels = [method_label(m, args.proposed_method, args.proposed_label) for m in methods]
    fig, ax = plt.subplots(figsize=(8.2, 1.25))
    ax.axis("off")
    handles = []
    for m, label in zip(methods, labels):
        line, = ax.plot(
            [], [], label=label,
            color=COLORS.get(label, COLORS.get(m, "#000000")),
            marker=MARKERS.get(label, MARKERS.get(m, "o")),
            linestyle="--", linewidth=1.5, markersize=5,
            markerfacecolor=COLORS.get(label, COLORS.get(m, "#000000")),
            markeredgecolor="black", markeredgewidth=0.5,
        )
        handles.append(line)
    ax.legend(handles=handles, loc="center", ncol=6, frameon=True, fancybox=False,
              shadow=False, edgecolor="black", fontsize=9, columnspacing=0.8,
              handlelength=1.5, handletextpad=0.5)
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"Runtime_Synthetic_Scale_Baseline_Legend.{ext}")
        plt.savefig(path, format=ext if ext == "pdf" else None, dpi=600, bbox_inches="tight")
    plt.close()


def plot_runtime_comparison(condition_summary: pd.DataFrame, methods: List[str], output_dir: str, args,
                            file_prefix: str = "Runtime_Synthetic_Scale_Baseline_Comparison") -> None:
    """
    Export one runtime figure per synthetic network model.

    This follows the SIR figure style: one 3.5 x 2.8 inch panel, model name on top,
    standalone legend, and PDF/PNG output for each model.
    """
    if condition_summary.empty:
        return

    models = [m for m in parse_csv_list(args.models.upper()) if m in set(condition_summary["NetworkModel"])]
    if not models:
        models = list(condition_summary["NetworkModel"].dropna().unique())

    plot_methods = parse_csv_list(args.plot_methods) if args.plot_methods else methods
    plot_methods = [m for m in plot_methods if m in methods]
    n_ticks = sorted(pd.to_numeric(condition_summary["N_Target"], errors="coerce").dropna().unique())

    for model in models:
        sub_model = condition_summary[condition_summary["NetworkModel"] == model].copy()
        if sub_model.empty:
            continue

        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        for method in plot_methods:
            label = method_label(method, args.proposed_method, args.proposed_label)
            sub = sub_model[sub_model["Method"] == method].sort_values("N_Target")
            if sub.empty:
                continue
            x = pd.to_numeric(sub["N_Target"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[args.plot_stat], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(sub["Runtime_CI95_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if not np.any(valid):
                continue
            lw = 1.6 if method == args.proposed_method else 1.2
            ms = 4.5 if method == args.proposed_method else 3.8
            zorder = 10 if method == args.proposed_method else 5
            ax.errorbar(
                x[valid], y[valid], yerr=yerr[valid],
                color=COLORS.get(label, COLORS.get(method, "#000000")),
                marker=MARKERS.get(label, MARKERS.get(method, "o")),
                linestyle="--", linewidth=lw, markersize=ms,
                markerfacecolor=COLORS.get(label, COLORS.get(method, "#000000")),
                markeredgecolor="black", markeredgewidth=0.5,
                ecolor=COLORS.get(label, COLORS.get(method, "#000000")),
                elinewidth=0.6, capsize=1.2, alpha=0.92,
                zorder=zorder,
            )

        ax.set_title(format_model_title(model), fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Network size")
        ax.set_ylabel("Running time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Keep the x-axis labels compact and consistent with the manuscript style:
        # only show 10^3 and 10^4 on the network-size axis.
        x_values = pd.to_numeric(sub_model["N_Target"], errors="coerce").dropna().to_numpy(dtype=float)
        if x_values.size > 0:
            x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
            major_x_ticks = [tick for tick in (1e3, 1e4) if x_min <= tick <= x_max]
            if not major_x_ticks:
                major_x_ticks = [10 ** int(np.floor(np.log10(x_min)))]
            ax.xaxis.set_major_locator(FixedLocator(major_x_ticks))
        else:
            ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4]))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.tick_params(axis="x", rotation=0)
        style_axes(ax)

        plt.tight_layout(pad=0.2)
        base = os.path.join(output_dir, f"{file_prefix}_{model}")
        plt.savefig(base + ".pdf", format="pdf")
        plt.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        print(f"[Output] Figure saved: {base}.pdf")


# =============================================================================
# 5. Main workflow
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic runtime scale comparison with baselines at fixed average degree.")
    parser.add_argument("--output-dir", default="results/exp_synthetic_scale_baseline_fixed_avg_degree_optimized_msh")
    parser.add_argument("--models", default="ER,BA,WS", help="Comma-separated synthetic models: ER,BA,WS.")
    parser.add_argument("--sizes", default="1000,2000,5000,10000,20000", help="Comma-separated N values.")
    parser.add_argument("--avg-degree", type=float, default=6.0, help="Target average degree.")
    parser.add_argument("--ws-rewiring", type=float, default=0.30, help="WS rewiring probability.")
    parser.add_argument("--instances", type=int, default=5, help="Independent graph instances per condition.")
    parser.add_argument("--repeat-times", type=int, default=1, help="Timing repeats per graph instance.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated method names.")
    parser.add_argument("--plot-methods", default="", help="Comma-separated subset of methods to plot; empty means all methods.")
    parser.add_argument("--plot-stat", default="Runtime_Mean_s", choices=["Runtime_Mean_s", "Runtime_MedianAcrossInstances_s"], help="Runtime statistic used in the figure.")
    parser.add_argument("--proposed-method", default="HOSH")
    parser.add_argument("--proposed-label", default="MSH")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--no-clean-output", action="store_true", help="Do not remove old workbook files before starting this run.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing runtime summaries and recompute all timings.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw figures from existing summary data; fail if no summary data are found.")
    parser.add_argument("--allow-unoptimized-fallback", action="store_true", help="Allow the script to run with the non-optimized hosh_methods.py implementation. By default, the experiment stops if the optimized MSH implementation is not detected.")
    parser.add_argument("--copy-to-mnt", action="store_true", help="After completion, also copy final outputs to /mnt/data for direct download in this environment.")
    args = parser.parse_args()

    models = [m.strip().upper() for m in parse_csv_list(args.models)]
    sizes = parse_int_list(args.sizes)
    methods = parse_csv_list(args.methods)

    if args.proposed_method not in methods:
        methods = [args.proposed_method] + methods

    os.makedirs(args.output_dir, exist_ok=True)

    # Default behavior: if a runtime summary already exists, redraw the figures directly
    # instead of rerunning the expensive timing experiment. Use --force-rerun to recompute.
    if not args.force_rerun:
        existing_summary = load_existing_condition_summary(args.output_dir, methods, args)
        if existing_summary is not None:
            export_standalone_legend(methods, args.output_dir, args)
            plot_runtime_comparison(existing_summary, methods, args.output_dir, args)
            print("=" * 80)
            print("Completed plotting from existing runtime data.")
            print(f"Output directory: {args.output_dir}")
            print("=" * 80)
            return
        if args.plot_only:
            raise FileNotFoundError(
                "--plot-only was specified, but no existing runtime summary was found in the output directory."
            )

    if not OPTIMIZED_MSH_ACTIVE and not args.allow_unoptimized_fallback:
        raise RuntimeError(
            "Optimized MSH/HOSH implementation was not detected. "
            "Place hosh_methods_fast_final.py in the project folder, or replace hosh_methods.py "
            "with the optimized implementation. Use --allow-unoptimized-fallback only for debugging."
        )

    checkpoint_path = os.path.join(args.output_dir, "Runtime_Synthetic_Scale_Baseline_Comparison_Checkpoint.xlsx")
    final_path = os.path.join(args.output_dir, "Runtime_Synthetic_Scale_Baseline_Comparison.xlsx")

    if not args.no_clean_output:
        remove_previous_outputs(args.output_dir, [
            "Runtime_Synthetic_Scale_Baseline_Comparison_Checkpoint.xlsx",
            "Runtime_Synthetic_Scale_Baseline_Comparison.xlsx",
            "Runtime_Synthetic_Scale_Baseline_Raw.csv",
            "Runtime_Synthetic_Scale_Baseline_Instance_Summary.csv",
            "Runtime_Synthetic_Scale_Baseline_Method_Summary.csv",
            "Runtime_Synthetic_Scale_Baseline_Graph_Summary.csv",
        ])

    notes = (
        "Synthetic scale-comparison runtime experiment. Network size N increases while the target average degree is fixed. "
        "Graph generation is excluded. Every method is run from scratch on the same preprocessed graph instance. "
        "HOSH is labeled as MSH in figures/tables. Existing runtime summary data are reused for plotting by default when available; use --force-rerun to recompute. No cached ranking or clique data is used during timing recomputation. The optimized exact MSH/HOSH implementation is required by default. It is loaded from hosh_methods_fast_final.py, hosh_methods_optimized.py, or an optimized replacement of hosh_methods.py."
    )
    export_standalone_legend(methods, args.output_dir, args)

    raw_rows: List[Dict[str, object]] = []
    instance_summary_rows: List[Dict[str, object]] = []
    graph_rows: List[Dict[str, object]] = []

    print("=" * 80)
    print("Synthetic runtime scale comparison with baselines")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"N values: {sizes}")
    print(f"Target average degree: {args.avg_degree}")
    print(f"Methods: {methods}")
    print(f"Instances: {args.instances}, repeat-times: {args.repeat_times}, master seed: {args.master_seed}")
    print(f"Method implementation source: {METHOD_IMPLEMENTATION_SOURCE}")
    print(f"Optimized MSH active: {OPTIMIZED_MSH_ACTIVE}")
    print("=" * 80)

    set_seed(args.master_seed)

    for model in models:
        print(f"\n[Model] {model}")
        for n_target in sizes:
            print(f"  [Condition] N={n_target}, <k>≈{args.avg_degree}")
            for instance in tqdm(range(args.instances), desc=f"    {model} N={n_target}", leave=False):
                inst_raw, inst_summary, inst_graph = run_one_instance(
                    model=model,
                    n_target=n_target,
                    avg_degree=args.avg_degree,
                    ws_rewiring=args.ws_rewiring,
                    instance=instance,
                    methods=methods,
                    args=args,
                )
                raw_rows.extend(inst_raw)
                instance_summary_rows.extend(inst_summary)
                graph_rows.append(inst_graph)

            # Save checkpoint after each model-size condition, following the revised experiment policy.
            save_workbook(raw_rows, instance_summary_rows, graph_rows, methods, args, checkpoint_path, notes)
            export_csv_outputs(args.output_dir, raw_rows, instance_summary_rows, graph_rows, methods, args)
            condition_summary = build_condition_summary(pd.DataFrame(instance_summary_rows), methods, args)
            plot_runtime_comparison(condition_summary, methods, args.output_dir, args)

    save_workbook(raw_rows, instance_summary_rows, graph_rows, methods, args, final_path, notes)
    export_csv_outputs(args.output_dir, raw_rows, instance_summary_rows, graph_rows, methods, args)
    condition_summary = build_condition_summary(pd.DataFrame(instance_summary_rows), methods, args)
    plot_runtime_comparison(condition_summary, methods, args.output_dir, args)

    if args.copy_to_mnt:
        filenames = [
            "Runtime_Synthetic_Scale_Baseline_Comparison.xlsx",
            "Runtime_Synthetic_Scale_Baseline_Method_Summary.csv",
            "Runtime_Synthetic_Scale_Baseline_Raw.csv",
        ]
        for model in models:
            filenames.extend([
                f"Runtime_Synthetic_Scale_Baseline_Comparison_{model}.pdf",
                f"Runtime_Synthetic_Scale_Baseline_Comparison_{model}.png",
            ])
        for filename in filenames:
            src = os.path.join(args.output_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("/mnt/data", filename))

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Final workbook: {final_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
