#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSH-only average-degree stress test with classic maximal-clique enumeration and 95% CIs.

Purpose
-------
This experiment evaluates the MSH runtime pipeline under increasing average
network degree while keeping network size fixed. It uses ER, BA, and WS
synthetic networks so that the effect of density can be examined under random,
scale-free, and clustered small-world topologies.

The reported runtime is decomposed into:
1) preprocessing time;
2) maximal-clique enumeration time using a selected exact backend;
3) MSH score-computation time.

Fresh-run policy
----------------
No precomputed rankings, cached scores, saved clique lists, or previous result
workbooks are read. Each run recomputes the full pipeline from the raw generated
graph.

Default protocol
----------------
Models: ER, BA, WS
N: 5000
Target average degrees: 4, 8, 12, 16, 20, 24
WS rewiring probability: 0.30
Independent graph instances per condition: 5
Timing repeats per instance: 1

Main manuscript table columns
-----------------------------
Model, target average degree, measured average degree, average clustering
coefficient, degeneracy, number of maximal cliques, maximum maximal-clique size, mean and maximum node-clique membership,
preprocessing time, maximal-clique enumeration time, and MSH score-computation
time. All summary values are reported as mean ± 95% confidence interval over
independent graph instances. If multiple timing repeats are used for the same
instance, repeats are averaged first and the CI is then computed across instances.

By default, maximal cliques are enumerated by NetworkX find_cliques, which is a
standard exact Bron-Kerbosch style algorithm with pivoting. The script therefore
avoids the faster igraph C backend unless --clique-backend igraph or auto is
explicitly specified.
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
import time
import traceback
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterSciNotation, NullFormatter
from tqdm import tqdm

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None


# =============================================================================
# 0. Plot configuration: SIR-style figures
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
    "savefig.bbox": None,
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

# =============================================================================
# 1. Reproducible utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(*items: object, modulus: int = 2**32 - 1) -> int:
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % modulus


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip().upper() for x in str(text).split(",") if x.strip()]


def ci95(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(arr)
    if n <= 1:
        return 0.0
    critical = float(student_t.ppf(0.975, df=n - 1)) if student_t is not None else 1.96
    return float(critical * np.std(arr, ddof=1) / math.sqrt(n))


def mean_val(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def fmt_mean_ci(mean_value: float, ci_value: float, decimals: int = 3) -> str:
    if not np.isfinite(mean_value):
        return ""
    if not np.isfinite(ci_value):
        ci_value = 0.0
    return f"{mean_value:.{decimals}f} ± {ci_value:.{decimals}f}"


def fmt_intlike_mean_ci(mean_value: float, ci_value: float) -> str:
    if not np.isfinite(mean_value):
        return ""
    if not np.isfinite(ci_value):
        ci_value = 0.0
    return f"{mean_value:.1f} ± {ci_value:.1f}"


def format_model_title(model: str) -> str:
    return str(model).upper()


def finish_axes_style(ax) -> None:
    """Use a uniform axes box so all four borders have identical visual weight."""
    # Hide native spines and draw one explicit rectangle. This avoids small
    # rendering differences among left/right/top/bottom spines in dense panels.
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)

    for patch in list(ax.patches):
        try:
            if patch.get_gid() == "uniform_axes_box":
                patch.remove()
        except Exception:
            pass

    rect = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax.transAxes,
        fill=False,
        edgecolor="#000000",
        linewidth=0.8,
        clip_on=False,
        zorder=1000,
    )
    rect.set_gid("uniform_axes_box")
    ax.add_patch(rect)

    ax.tick_params(direction="out", which="major", length=3.0, width=0.7, top=False, right=False)
    ax.tick_params(direction="out", which="minor", length=2.0, width=0.6, top=False, right=False)


def safe_log_xerr(mean_values, ci_values):
    """Create asymmetric x-errors that remain positive on a log-scaled x-axis."""
    mean_values = np.asarray(mean_values, dtype=float)
    ci_values = np.asarray(ci_values, dtype=float)
    ci_values = np.where(np.isfinite(ci_values), ci_values, 0.0)
    lower = np.minimum(ci_values, np.maximum(mean_values * 0.90, 1e-12))
    upper = ci_values
    return np.vstack([lower, upper])


def find_existing_summary(output_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load an existing summary table if available, without recomputing the experiment."""
    candidates = [
        os.path.join(output_dir, "MSH_Density_Stress_Summary_Mean_CI.csv"),
        os.path.join(output_dir, "MSH_Density_Stress_Table.xlsx"),
        os.path.join(output_dir, "MSH_Density_Stress_Table_Checkpoint.xlsx"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, sheet_name="Summary_Mean_CI")
            if not df.empty:
                return df, path
        except Exception as exc:
            print(f"[Warning] Failed to read existing summary from {path}: {exc}")
    return pd.DataFrame(), ""


# =============================================================================
# 2. Synthetic graph generation and preprocessing
# =============================================================================
def make_er_graph(n: int, avg_degree: float, seed: int) -> nx.Graph:
    p = min(max(float(avg_degree) / max(n - 1, 1), 0.0), 1.0)
    return nx.erdos_renyi_graph(n, p, seed=seed)


def make_ba_graph(n: int, avg_degree: float, seed: int) -> nx.Graph:
    # BA average degree is approximately 2m. Therefore odd target degrees are
    # rounded to the nearest feasible even-degree regime.
    m = max(1, int(round(float(avg_degree) / 2.0)))
    m = min(m, max(1, n - 1))
    return nx.barabasi_albert_graph(n, m, seed=seed)


def make_ws_graph(n: int, avg_degree: float, rewiring_prob: float, seed: int) -> nx.Graph:
    # NetworkX WS requires an even k.
    k = int(round(float(avg_degree)))
    if k < 2:
        k = 2
    if k % 2 == 1:
        k += 1
    k = min(k, n - 1)
    if k % 2 == 1:
        k -= 1
    k = max(2, k)
    return nx.watts_strogatz_graph(n, k, rewiring_prob, seed=seed)


def generate_graph(model: str, n: int, avg_degree: float, instance: int, master_seed: int, ws_rewiring: float) -> nx.Graph:
    seed = stable_int_hash("msh-density-stress", model, n, avg_degree, instance, master_seed, ws_rewiring)
    if model == "ER":
        return make_er_graph(n, avg_degree, seed)
    if model == "BA":
        return make_ba_graph(n, avg_degree, seed)
    if model == "WS":
        return make_ws_graph(n, avg_degree, ws_rewiring, seed)
    raise ValueError(f"Unknown model: {model}")


def preprocess_graph(g_raw: nx.Graph) -> nx.Graph:
    g = nx.Graph(g_raw)
    g.remove_edges_from(nx.selfloop_edges(g))
    if g.number_of_nodes() == 0:
        return g
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")


# =============================================================================
# 3. Maximal cliques and MSH scoring
# =============================================================================
def enumerate_maximal_cliques_only(g: nx.Graph, backend_choice: str = "networkx") -> Tuple[List[List[int]], str]:
    """List maximal cliques only. Incidence construction is counted as score computation.

    Parameters
    ----------
    backend_choice:
        - "networkx": use NetworkX find_cliques. This is the default and is the
          conservative classic exact Bron-Kerbosch style backend.
        - "igraph": use igraph maximal_cliques. This is faster because the
          implementation is in C, but it is not used by default here.
        - "auto": try igraph first and fall back to NetworkX.
    """
    backend_choice = str(backend_choice).lower().strip()
    if backend_choice not in {"networkx", "igraph", "auto"}:
        raise ValueError(f"Unknown clique backend: {backend_choice}")

    if backend_choice in {"igraph", "auto"}:
        try:
            import igraph as ig  # type: ignore
            g_ig = ig.Graph(n=g.number_of_nodes(), edges=list(g.edges()), directed=False)
            return [list(c) for c in g_ig.maximal_cliques()], "igraph.maximal_cliques"
        except Exception:
            if backend_choice == "igraph":
                raise

    return [list(c) for c in nx.find_cliques(g)], "networkx.find_cliques"


def build_clique_data(g: nx.Graph, cliques: List[List[int]]) -> Dict[str, object]:
    node_cliques_map: Dict[int, List[int]] = {n: [] for n in g.nodes()}
    for idx, clique in enumerate(cliques):
        for node in clique:
            node_cliques_map[node].append(idx)
    clique_sizes = [len(c) for c in cliques]
    return {
        "cliques": cliques,
        "node_cliques_map": node_cliques_map,
        "clique_sizes": clique_sizes,
    }


def build_directed_overlap_cache(node_cliques_map: Dict[int, List[int]], clique_sizes: List[int]) -> Dict[Tuple[int, int], float]:
    pair_shared_count: Dict[Tuple[int, int], int] = {}
    for clique_indices in node_cliques_map.values():
        L = len(clique_indices)
        if L <= 1:
            continue
        for i in range(L - 1):
            a = clique_indices[i]
            if clique_sizes[a] <= 2:
                continue
            for j in range(i + 1, L):
                b = clique_indices[j]
                if clique_sizes[b] <= 2:
                    continue
                key = (a, b) if a < b else (b, a)
                pair_shared_count[key] = pair_shared_count.get(key, 0) + 1

    directed_overlap: Dict[Tuple[int, int], float] = {}
    for (a, b), shared_size in pair_shared_count.items():
        numerator = shared_size - 1
        if numerator <= 0:
            continue
        denom_a = clique_sizes[a] - 1
        denom_b = clique_sizes[b] - 1
        if denom_a > 0:
            directed_overlap[(a, b)] = numerator / denom_a
        if denom_b > 0:
            directed_overlap[(b, a)] = numerator / denom_b
    return directed_overlap


def calculate_msh_scores(g: nx.Graph, cliques: List[List[int]], xi: float = 0.001) -> Tuple[Dict[int, float], Dict[str, float], Dict[str, object]]:
    """Compute MSH scores from already enumerated cliques.

    The returned score-computation time includes clique-incidence construction,
    external capability, effective dependence, overlap-cache construction, final
    constraint aggregation, and final score sorting. The clique-listing time is
    measured separately outside this function.
    """
    t_score_start = time.perf_counter()

    t0 = time.perf_counter()
    clique_data = build_clique_data(g, cliques)
    incidence_time = time.perf_counter() - t0

    node_cliques_map: Dict[int, List[int]] = clique_data["node_cliques_map"]  # type: ignore[assignment]
    clique_sizes: List[int] = clique_data["clique_sizes"]  # type: ignore[assignment]
    nodes = list(g.nodes())
    degrees = dict(g.degree())

    # External capability.
    t0 = time.perf_counter()
    clique_node_k_totals: List[Dict[int, float]] = [None] * len(cliques)  # type: ignore[list-item]
    clique_norm_denominators: List[float] = [xi] * len(cliques)
    for alpha, c_nodes in enumerate(cliques):
        c_size = clique_sizes[alpha]
        base = c_size - 1
        ext_values: List[Tuple[int, float]] = []
        total_ext = 0.0
        for node in c_nodes:
            val = max(0.0, float(degrees[node] - base))
            ext_values.append((node, val))
            total_ext += val
        k_map: Dict[int, float] = {}
        max_k = 0.0
        if c_size > 1:
            for node, node_ext in ext_values:
                k_val = node_ext + math.log1p(total_ext - node_ext)
                k_map[node] = k_val
                if k_val > max_k:
                    max_k = k_val
        else:
            for node, node_ext in ext_values:
                k_map[node] = node_ext
                if node_ext > max_k:
                    max_k = node_ext
        clique_node_k_totals[alpha] = k_map
        clique_norm_denominators[alpha] = max_k + xi
    external_capability_time = time.perf_counter() - t0

    # Effective dependence.
    t0 = time.perf_counter()
    node_p_stars: Dict[int, Dict[int, float]] = {}
    for v in nodes:
        my_indices = node_cliques_map.get(v, [])
        if not my_indices:
            continue
        p_base = 1.0 / len(my_indices)
        p_dict: Dict[int, float] = {}
        for alpha in my_indices:
            k_total_i = clique_node_k_totals[alpha][v]
            p_dict[alpha] = p_base * (1.0 - k_total_i / clique_norm_denominators[alpha])
        node_p_stars[v] = p_dict
    effective_dependence_time = time.perf_counter() - t0

    # Overlap cache.
    t0 = time.perf_counter()
    directed_overlap = build_directed_overlap_cache(node_cliques_map, clique_sizes)
    overlap_by_beta: Dict[int, List[Tuple[int, float]]] = {}
    for (beta, alpha), weight in directed_overlap.items():
        overlap_by_beta.setdefault(beta, []).append((alpha, weight))
    overlap_cache_time = time.perf_counter() - t0

    # Final constraint aggregation.
    t0 = time.perf_counter()
    scores: Dict[int, float] = {n: 0.0 for n in nodes}
    for v in nodes:
        my_indices = node_cliques_map.get(v, [])
        if not my_indices:
            continue
        p_dict = node_p_stars[v]
        my_set = set(my_indices)
        indirect_by_alpha = {alpha: 0.0 for alpha in my_indices}
        for beta, p_beta in p_dict.items():
            for alpha, weight in overlap_by_beta.get(beta, []):
                if alpha in my_set:
                    indirect_by_alpha[alpha] += p_beta * weight
        total_constraint = 0.0
        for alpha in my_indices:
            val = p_dict[alpha] + indirect_by_alpha[alpha]
            total_constraint += val * val
        scores[v] = 1.0 - total_constraint
    score_aggregation_time = time.perf_counter() - t0

    # Sorting is retained in score-computation time because the method output is a ranking.
    t0 = time.perf_counter()
    _ = sorted(scores, key=lambda node: (-round(float(scores[node]), 8), node))
    ranking_sort_time = time.perf_counter() - t0

    score_time_total = time.perf_counter() - t_score_start
    timing = {
        "Clique_Incidence_Build_Time_s": incidence_time,
        "External_Capability_Time_s": external_capability_time,
        "Effective_Dependence_Time_s": effective_dependence_time,
        "Overlap_Cache_Time_s": overlap_cache_time,
        "Score_Aggregation_Time_s": score_aggregation_time,
        "Ranking_Sort_Time_s": ranking_sort_time,
        "Score_Computation_Time_s": score_time_total,
    }
    return scores, timing, clique_data


# =============================================================================
# 4. Structural statistics and one full run
# =============================================================================
def safe_degeneracy(g: nx.Graph) -> int:
    if g.number_of_nodes() == 0:
        return 0
    try:
        core = nx.core_number(g)
        return int(max(core.values())) if core else 0
    except Exception:
        return 0


def structure_stats(g: nx.Graph, clique_data: Dict[str, object]) -> Dict[str, float]:
    clique_sizes = list(clique_data["clique_sizes"])  # type: ignore[arg-type]
    node_cliques_map = clique_data.get("node_cliques_map", {})
    n_cliques = len(clique_sizes)
    arr = np.asarray(clique_sizes, dtype=float) if n_cliques else np.asarray([], dtype=float)
    memberships = np.asarray([len(node_cliques_map.get(n, [])) for n in g.nodes()], dtype=float) if g.number_of_nodes() else np.asarray([], dtype=float)
    return {
        "Processed_N": int(g.number_of_nodes()),
        "Processed_E": int(g.number_of_edges()),
        "Actual_Avg_Degree": float(2.0 * g.number_of_edges() / g.number_of_nodes()) if g.number_of_nodes() else 0.0,
        "Average_Clustering": float(nx.average_clustering(g)) if g.number_of_nodes() else np.nan,
        "Transitivity": float(nx.transitivity(g)) if g.number_of_nodes() else np.nan,
        "Degeneracy": safe_degeneracy(g),
        "Maximal_Clique_Count": int(n_cliques),
        "Max_Clique_Size": int(arr.max()) if n_cliques else 0,
        "Mean_Clique_Size": float(arr.mean()) if n_cliques else 0.0,
        "Mean_Clique_Membership_Per_Node": float(memberships.mean()) if memberships.size else 0.0,
        "Max_Clique_Membership_Per_Node": int(memberships.max()) if memberships.size else 0,
    }


def run_full_pipeline_once(g_raw: nx.Graph, run_seed: int, clique_backend: str = "networkx") -> Dict[str, object]:
    set_seed(run_seed)
    gc.collect()

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    g = preprocess_graph(g_raw)
    preprocessing_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cliques, backend = enumerate_maximal_cliques_only(g, backend_choice=clique_backend)
    clique_enum_time = time.perf_counter() - t0

    _, score_timing, clique_data = calculate_msh_scores(g, cliques)
    total_time = time.perf_counter() - total_start

    rec: Dict[str, object] = {
        "Preprocessing_Time_s": preprocessing_time,
        "Clique_Enumeration_Time_s": clique_enum_time,
        "Score_Computation_Time_s": score_timing["Score_Computation_Time_s"],
        "Total_MSH_Time_s": total_time,
        "Clique_Backend": backend,
    }
    rec.update(score_timing)
    rec.update(structure_stats(g, clique_data))
    return rec


# =============================================================================
# 5. Summary tables and export
# =============================================================================
def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    group_cols = ["Model", "Target_Avg_Degree"]
    instance_cols = group_cols + ["Instance"]
    numeric_cols = [
        "Raw_N", "Raw_E", "Processed_N", "Processed_E", "Actual_Avg_Degree",
        "Average_Clustering", "Transitivity", "Degeneracy", "Maximal_Clique_Count",
        "Max_Clique_Size", "Mean_Clique_Size", "Mean_Clique_Membership_Per_Node", "Max_Clique_Membership_Per_Node", "Preprocessing_Time_s",
        "Clique_Enumeration_Time_s", "Score_Computation_Time_s", "Total_MSH_Time_s",
        "Clique_Incidence_Build_Time_s", "External_Capability_Time_s",
        "Effective_Dependence_Time_s", "Overlap_Cache_Time_s",
        "Score_Aggregation_Time_s", "Ranking_Sort_Time_s",
    ]
    numeric_cols = [c for c in numeric_cols if c in raw_df.columns]

    # Average repeated timings on the same generated graph instance first.
    instance_df = (
        raw_df[instance_cols + numeric_cols]
        .groupby(instance_cols, dropna=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for key, sub in instance_df.groupby(group_cols, dropna=False):
        row: Dict[str, object] = dict(zip(group_cols, key))
        row["Instances"] = int(sub["Instance"].nunique())
        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_Mean"] = mean_val(vals)
            row[f"{col}_CI95"] = ci95(vals)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["Model", "Target_Avg_Degree"]).reset_index(drop=True)
    return out


def build_manuscript_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            "Model": row["Model"],
            "Target <k>": f"{float(row['Target_Avg_Degree']):g}",
            "Actual <k>": fmt_mean_ci(row["Actual_Avg_Degree_Mean"], row["Actual_Avg_Degree_CI95"], 2),
            "Avg. clustering": fmt_mean_ci(row["Average_Clustering_Mean"], row["Average_Clustering_CI95"], 3),
            "Degeneracy": fmt_intlike_mean_ci(row["Degeneracy_Mean"], row["Degeneracy_CI95"]),
            "# maximal cliques": fmt_intlike_mean_ci(row["Maximal_Clique_Count_Mean"], row["Maximal_Clique_Count_CI95"]),
            "Max clique size": fmt_intlike_mean_ci(row["Max_Clique_Size_Mean"], row["Max_Clique_Size_CI95"]),
            "Mean membership": fmt_mean_ci(row["Mean_Clique_Membership_Per_Node_Mean"], row["Mean_Clique_Membership_Per_Node_CI95"], 2),
            "Max membership": fmt_intlike_mean_ci(row["Max_Clique_Membership_Per_Node_Mean"], row["Max_Clique_Membership_Per_Node_CI95"]),
            "Preprocessing time (s)": fmt_mean_ci(row["Preprocessing_Time_s_Mean"], row["Preprocessing_Time_s_CI95"], 4),
            "Clique enumeration time (s)": fmt_mean_ci(row["Clique_Enumeration_Time_s_Mean"], row["Clique_Enumeration_Time_s_CI95"], 4),
            "Score computation time (s)": fmt_mean_ci(row["Score_Computation_Time_s_Mean"], row["Score_Computation_Time_s_CI95"], 4),
        })
    return pd.DataFrame(rows)


def notes_df(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        ("Generated at", _dt.datetime.now().isoformat(timespec="seconds")),
        ("Experiment", "MSH-only average-degree stress test"),
        ("Figure lower panel", "Three raw structure subpanels for |M|, s_max, and degeneracy d; |M| uses a log-scaled y-axis."),
        ("Fresh-run policy", "No precomputed rankings, score caches, clique caches, or previous results are read."),
        ("Timing scope", "Graph generation is excluded. Preprocessing, maximal-clique listing, and MSH score computation are measured separately."),
        ("Clique enumeration backend", args.clique_backend),
        ("Score computation", "Includes clique-incidence construction, external capability, effective dependence, overlap-cache construction, final score aggregation, and ranking sort."),
        ("Models", args.models),
        ("N", str(args.n)),
        ("Target average degrees", args.avg_degrees),
        ("WS rewiring probability", str(args.ws_rewiring)),
        ("Instances", str(args.instances)),
        ("Timing repeats per instance", str(args.repeat_times)),
        ("Master seed", str(args.master_seed)),
        ("Python", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("Processor", platform.processor()),
        ("NetworkX", getattr(nx, "__version__", "unknown")),
        ("NumPy", getattr(np, "__version__", "unknown")),
        ("Pandas", getattr(pd, "__version__", "unknown")),
    ]
    try:
        import igraph as ig  # type: ignore
        rows.append(("igraph", getattr(ig, "__version__", "available")))
    except Exception:
        rows.append(("igraph", "not available"))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def write_outputs(raw_records: List[Dict[str, object]], output_dir: str, args: argparse.Namespace, checkpoint: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_df = pd.DataFrame(raw_records)
    summary_df = summarize(raw_df) if not raw_df.empty else pd.DataFrame()
    manuscript_df = build_manuscript_table(summary_df) if not summary_df.empty else pd.DataFrame()
    suffix = "_Checkpoint" if checkpoint else ""
    xlsx_path = os.path.join(output_dir, f"MSH_Density_Stress_Table{suffix}.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        manuscript_df.to_excel(writer, sheet_name="Manuscript_Table", index=False)
        summary_df.to_excel(writer, sheet_name="Summary_Mean_CI", index=False)
        raw_df.to_excel(writer, sheet_name="Raw", index=False)
        notes_df(args).to_excel(writer, sheet_name="Protocol_Metadata", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) for c in col_cells if c.value is not None)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, max_len + 2), 46)

    if not checkpoint:
        raw_df.to_csv(os.path.join(output_dir, "MSH_Density_Stress_Raw.csv"), index=False, encoding="utf-8-sig")
        summary_df.to_csv(os.path.join(output_dir, "MSH_Density_Stress_Summary_Mean_CI.csv"), index=False, encoding="utf-8-sig")
        manuscript_df.to_csv(os.path.join(output_dir, "MSH_Density_Stress_Manuscript_Table.csv"), index=False, encoding="utf-8-sig")
        tex_path = os.path.join(output_dir, "MSH_Density_Stress_Manuscript_Table.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(manuscript_df.to_latex(index=False, escape=False))
    print(f"[Output] {xlsx_path}")



# =============================================================================
# 5b. Diagnostic runtime-structure correlation and plotting
# =============================================================================
# Only the three clique-structure indicators used in the manuscript figure.
# They are shown as three raw-value subpanels rather than normalized curves.
STRUCTURE_METRIC_SPECS = [
    ("Maximal cliques", "Maximal_Clique_Count_Mean", r"$|\mathcal{M}|$", True),
    ("Max clique size", "Max_Clique_Size_Mean", r"$s_{max}$", False),
    ("Degeneracy", "Degeneracy_Mean", r"$d$", False),
]

RUNTIME_METRIC_SPECS = [
    ("T_enum", "Clique_Enumeration_Time_s_Mean"),
    ("T_score", "Score_Computation_Time_s_Mean"),
    ("T_total", "Total_MSH_Time_s_Mean"),
]


def _safe_corr(x, y, method: str = "pearson") -> float:
    sx = pd.to_numeric(pd.Series(x), errors="coerce")
    sy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = sx.notna() & sy.notna()
    sx = sx[mask]
    sy = sy[mask]
    if len(sx) < 3:
        return np.nan
    if sx.nunique() <= 1 or sy.nunique() <= 1:
        return np.nan
    return float(sx.corr(sy, method=method))


def _log10_positive(arr) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    mask = np.isfinite(vals) & (vals > 0)
    out[mask] = np.log10(vals[mask])
    return out


def _normalized_curve(values, log_transform: bool = False) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if log_transform:
        vals = _log10_positive(vals)
    out = np.full_like(vals, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return out
    vmin = float(np.nanmin(vals[mask]))
    vmax = float(np.nanmax(vals[mask]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        out[mask] = 0.5
    else:
        out[mask] = (vals[mask] - vmin) / (vmax - vmin)
    return out


def export_runtime_structure_correlations(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Export correlation diagnostics between runtime and structural indicators.

    The correlation table is intended for analysis and for deciding which clique
    statistics should be discussed in the manuscript. It is not necessarily the
    final manuscript table.
    """
    if summary_df is None or summary_df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    model_values = ["ALL"] + sorted(str(x).upper() for x in summary_df["Model"].dropna().unique())

    for model in model_values:
        if model == "ALL":
            sub = summary_df.copy()
        else:
            sub = summary_df[summary_df["Model"].astype(str).str.upper() == model].copy()
        if sub.empty:
            continue

        for rt_label, rt_col in RUNTIME_METRIC_SPECS:
            if rt_col not in sub.columns:
                continue
            y = pd.to_numeric(sub[rt_col], errors="coerce")
            for metric_label, metric_col, _, _ in STRUCTURE_METRIC_SPECS:
                if metric_col not in sub.columns:
                    continue
                x = pd.to_numeric(sub[metric_col], errors="coerce")
                rows.append({
                    "Model": model,
                    "Runtime": rt_label,
                    "Structure_metric": metric_label,
                    "Pearson": _safe_corr(x, y, method="pearson"),
                    "Spearman": _safe_corr(x, y, method="spearman"),
                    "Log_Pearson": _safe_corr(_log10_positive(x), _log10_positive(y), method="pearson"),
                    "N": int((x.notna() & y.notna()).sum()),
                })

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return

    corr_df["Abs_Spearman"] = corr_df["Spearman"].abs()
    top_df = corr_df.sort_values(["Model", "Runtime", "Abs_Spearman"], ascending=[True, True, False])

    csv_path = os.path.join(output_dir, "MSH_Density_Stress_Runtime_Structure_Correlation.csv")
    xlsx_path = os.path.join(output_dir, "MSH_Density_Stress_Runtime_Structure_Correlation.xlsx")
    corr_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        corr_df.to_excel(writer, sheet_name="Correlation", index=False)
        top_df.to_excel(writer, sheet_name="Sorted_by_abs_Spearman", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) for c in col_cells if c.value is not None)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, max_len + 2), 34)
    print(f"[Output] Correlation diagnostics saved: {xlsx_path}")


def plot_density_grouped_bar(summary_df: pd.DataFrame, output_dir: str, args: argparse.Namespace) -> None:
    """Export one runtime-plus-raw-structure figure per network model.

    The figure follows the two-level manuscript style used in the previous
    runtime-clique-structure plots, but the structure part is split into three
    small raw-value panels: |M|, s_max, and d. This avoids normalization and
    avoids a dual y-axis while preserving the original values.
    """
    if summary_df is None or summary_df.empty:
        print("[Plot] Empty summary table; no figure generated.")
        return

    required = [
        "Model", "Target_Avg_Degree",
        "Clique_Enumeration_Time_s_Mean", "Clique_Enumeration_Time_s_CI95",
        "Score_Computation_Time_s_Mean", "Score_Computation_Time_s_CI95",
        "Maximal_Clique_Count_Mean", "Maximal_Clique_Count_CI95",
        "Max_Clique_Size_Mean", "Max_Clique_Size_CI95",
        "Degeneracy_Mean", "Degeneracy_CI95",
    ]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        print(f"[Plot] Missing required columns: {missing}")
        return

    export_runtime_structure_correlations(summary_df, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    models = parse_str_list(args.models)
    available = set(str(x).upper() for x in summary_df["Model"].dropna().unique())
    models = [m for m in models if m in available] or sorted(available)

    # Same visual language as the previous runtime-clique-structure figure.
    bar_specs = [
        (r"$T_{enum}$", "Clique_Enumeration_Time_s_Mean", "Clique_Enumeration_Time_s_CI95", "#68ADD3", -0.19),
        (r"$T_{score}$", "Score_Computation_Time_s_Mean", "Score_Computation_Time_s_CI95", "#D94A48", 0.19),
    ]
    struct_specs = [
        (r"$|\mathcal{M}|$", "Maximal_Clique_Count_Mean", "Maximal_Clique_Count_CI95", "log", "#4D4D4D", "o", "--"),
        (r"$s_{max}$", "Max_Clique_Size_Mean", "Max_Clique_Size_CI95", "linear", "#D55E00", "s", ":"),
        (r"$d$", "Degeneracy_Mean", "Degeneracy_CI95", "linear", "#2A7F62", "^", "-."),
    ]

    def _positive_log_yerr(yv: np.ndarray, ev: np.ndarray, floor: float = 1e-12) -> np.ndarray:
        """Return asymmetric y-errors whose lower end stays above a positive floor.

        This is important for vector PDF output on log-scaled axes.  Error bars
        reaching zero or negative values can create extremely large transforms
        when the PDF is opened in Adobe Illustrator.
        """
        yv = np.asarray(yv, dtype=float)
        ev = np.asarray(ev, dtype=float)
        floor = max(float(floor), 1e-12)
        lower_cap = np.maximum(yv - floor * 1.02, 1e-12)
        lower = np.minimum(np.maximum(ev, 0.0), lower_cap)
        upper = np.maximum(ev, 0.0)
        return np.vstack([lower, upper])

    def _draw_log_safe_bars(ax, xpos: np.ndarray, height_values: np.ndarray, floor: float, width: float,
                            color: str, label: str) -> None:
        """Draw bars on a log y-axis without a zero baseline.

        Matplotlib bars default to bottom=0.  On a log y-axis this sends the
        lower rectangle vertices to -inf in display space, which can trigger
        Adobe Illustrator warnings such as requested transforms outside the
        drawing area.  A positive baseline keeps every vector object finite.
        """
        floor = max(float(floor), 1e-12)
        yv = np.asarray(height_values, dtype=float)
        bar_heights = np.maximum(yv - floor, floor * 1e-6)
        bars = ax.bar(
            xpos,
            bar_heights,
            bottom=floor,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.92,
            label=label,
            zorder=5,
        )
        for patch in bars.patches:
            patch.set_clip_on(True)

    def _linear_yerr(yv: np.ndarray, ev: np.ndarray) -> np.ndarray:
        lower = np.minimum(ev, np.maximum(yv, 0.0))
        upper = ev
        return np.vstack([lower, upper])

    def _set_linear_ylim(ax, yv: np.ndarray, ev: np.ndarray) -> None:
        if len(yv) == 0:
            return
        low = yv - np.minimum(ev, np.maximum(yv, 0.0))
        high = yv + ev
        ymin = float(np.nanmin(low))
        ymax = float(np.nanmax(high))
        span = ymax - ymin
        if not np.isfinite(span) or span <= 1e-12:
            span = max(1.0, abs(ymax) * 0.2)
        ax.set_ylim(max(0.0, ymin - 0.16 * span), ymax + 0.24 * span)

    def _set_log_ylim(ax, yv: np.ndarray, ev: np.ndarray) -> None:
        if len(yv) == 0:
            return
        lower = np.minimum(ev, np.maximum(yv * 0.90, 1e-12))
        positive_lower = np.maximum(yv - lower, 1e-12)
        upper_bound = yv + ev
        ymin = float(np.nanmin(positive_lower))
        ymax = float(np.nanmax(upper_bound))
        if ymin > 0 and ymax > ymin:
            ax.set_ylim(ymin / 1.55, ymax * 1.65)

    for model in models:
        sub = summary_df[summary_df["Model"].astype(str).str.upper() == model].copy()
        if sub.empty:
            continue
        sub["Target_Avg_Degree"] = pd.to_numeric(sub["Target_Avg_Degree"], errors="coerce")
        sub = sub.dropna(subset=["Target_Avg_Degree"]).sort_values("Target_Avg_Degree")
        if sub.empty:
            continue

        degrees = sub["Target_Avg_Degree"].to_numpy(dtype=float)
        x = np.arange(len(degrees), dtype=float)
        width = 0.34

        fig, axes = plt.subplots(
            4, 1,
            figsize=(3.5, 5.10),
            sharex=True,
            gridspec_kw={"height_ratios": [2.20, 0.72, 0.72, 0.72], "hspace": 0.12},
        )
        ax_time = axes[0]
        structure_axes = axes[1:]

        # --------------------------------------------------------------
        # Upper panel: runtime decomposition.
        # --------------------------------------------------------------
        runtime_plot_data = []
        all_runtime_values = []
        all_runtime_lower = []
        for label, mean_col, ci_col, color, offset in bar_specs:
            y = pd.to_numeric(sub[mean_col], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(sub[ci_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            valid = np.isfinite(y) & (y > 0)
            if not np.any(valid):
                continue
            yv = y[valid]
            ev = np.maximum(yerr[valid], 0.0)
            lower_tmp = np.minimum(ev, np.maximum(yv * 0.90, 1e-12))
            lower_end = yv - lower_tmp
            all_runtime_values.extend(yv.tolist())
            all_runtime_lower.extend(lower_end[lower_end > 0].tolist())
            runtime_plot_data.append((label, color, offset, valid, yv, ev))

        if all_runtime_lower:
            runtime_floor = max(min(all_runtime_lower) / 1.35, 1e-12)
        elif all_runtime_values:
            runtime_floor = max(min(all_runtime_values) / 3.0, 1e-12)
        else:
            runtime_floor = 1e-6

        ax_time.set_yscale("log")
        for label, color, offset, valid, yv, ev in runtime_plot_data:
            xpos = x[valid] + offset
            _draw_log_safe_bars(ax_time, xpos, yv, runtime_floor, width, color, label)
            err_container = ax_time.errorbar(
                xpos,
                yv,
                yerr=_positive_log_yerr(yv, ev, floor=runtime_floor),
                fmt="none",
                ecolor="black",
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                zorder=8,
            )
            # Keep all errorbar artists clipped to the axes; this also improves
            # compatibility with vector editors.
            for artist in list(err_container.lines):
                if artist is None:
                    continue
                if isinstance(artist, (tuple, list)):
                    for a in artist:
                        try:
                            a.set_clip_on(True)
                        except Exception:
                            pass
                else:
                    try:
                        artist.set_clip_on(True)
                    except Exception:
                        pass

        ax_time.set_title(format_model_title(model), fontsize=12, fontweight="bold", pad=6)
        ax_time.set_ylabel("Running time (s)", fontsize=10.5, labelpad=2)
        ax_time.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax_time.yaxis.set_minor_formatter(NullFormatter())
        if all_runtime_values:
            ymax = max(all_runtime_values)
            if runtime_floor > 0 and ymax > runtime_floor:
                ax_time.set_ylim(runtime_floor, ymax * 3.0)
        ax_time.legend(
            loc="upper left",
            frameon=True,
            fancybox=False,
            shadow=False,
            edgecolor="black",
            framealpha=1.0,
            fontsize=8.5,
            borderpad=0.25,
            handletextpad=0.35,
            labelspacing=0.18,
        )
        finish_axes_style(ax_time)

        # --------------------------------------------------------------
        # Lower three small panels: raw clique-structure indicators.
        # --------------------------------------------------------------
        for ax, (label, mean_col, ci_col, scale, color, marker, linestyle) in zip(structure_axes, struct_specs):
            y = pd.to_numeric(sub[mean_col], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(sub[ci_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            valid = np.isfinite(y)
            if scale == "log":
                valid = valid & (y > 0)
            if not np.any(valid):
                finish_axes_style(ax)
                continue

            xv = x[valid]
            yv = y[valid]
            ev = yerr[valid]
            if scale == "log":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
                ax.yaxis.set_minor_formatter(NullFormatter())
                yerr_plot = _positive_log_yerr(yv, ev)
                _set_log_ylim(ax, yv, ev)
            else:
                yerr_plot = _linear_yerr(yv, ev)
                _set_linear_ylim(ax, yv, ev)

            ax.errorbar(
                xv,
                yv,
                yerr=yerr_plot,
                fmt=linestyle + marker,
                color=color,
                linewidth=1.05,
                markersize=3.2,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
                ecolor=color,
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                zorder=6,
            )
            ax.set_ylabel(label, fontsize=10.5, labelpad=2)
            finish_axes_style(ax)

        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

        structure_axes[-1].set_xlabel(r"$\langle k\rangle$", fontsize=10.5, labelpad=3)
        structure_axes[-1].set_xticks(x)
        structure_axes[-1].set_xticklabels([f"{d:g}" for d in degrees])

        for ax in axes:
            ax.tick_params(axis="both", which="major", labelsize=9, width=0.7, length=3.0, top=False, right=False)
            ax.tick_params(axis="both", which="minor", width=0.6, length=2.0, top=False, right=False)
            finish_axes_style(ax)

        # Keep all y-axis labels on the same vertical line.  This avoids the
        # apparent misalignment caused by different tick-label widths on log
        # and linear axes.
        for ax in axes:
            ax.yaxis.set_label_coords(-0.145, 0.5)
        try:
            fig.align_ylabels(axes)
        except Exception:
            pass
        fig.subplots_adjust(left=0.245, right=0.985, top=0.945, bottom=0.090, hspace=0.12)

        base = os.path.join(output_dir, f"MSH_Density_Stress_Runtime_StyleThreeSubpanels_AISafe_{model}")
        # Use a standard PDF bounding box.  The global rcParam previously used
        # tight bounding boxes, which can interact poorly with log-scale vector
        # rectangles in Illustrator.  PNG keeps a tight bounding box for preview.
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(base + ".pdf", format="pdf", bbox_inches=None)
        fig.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"[Output] Figure saved: {base}.pdf")

def clean_old_outputs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in [
        "MSH_Density_Stress_Table.xlsx",
        "MSH_Density_Stress_Table_Checkpoint.xlsx",
        "MSH_Density_Stress_Raw.csv",
        "MSH_Density_Stress_Summary_Mean_CI.csv",
        "MSH_Density_Stress_Manuscript_Table.csv",
        "MSH_Density_Stress_Manuscript_Table.tex",
    ]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# =============================================================================
# 6. Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="MSH-only average-degree stress test with manuscript-ready table and 95% confidence intervals.")
    parser.add_argument("--output-dir", default="results/exp_msh_density_stress_table_ci_avgdegree_4_24")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--models", default="ER,BA,WS")
    parser.add_argument("--avg-degrees", default="4,8,12,16,20,24")
    parser.add_argument("--ws-rewiring", type=float, default=0.30)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--clique-backend", choices=["networkx", "igraph", "auto"], default="networkx", help="Backend for maximal-clique enumeration. Default networkx uses the classic exact Bron-Kerbosch style backend and avoids the faster igraph C backend.")
    parser.add_argument("--no-clean-output", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Only read existing summary data and redraw grouped-bar figures.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing summary data and recompute the full experiment.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate grouped-bar figures after the table outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # By default, reuse existing summary data for plotting if available.
    # Use --force-rerun when generating formal fresh-run results.
    if not args.force_rerun:
        existing_summary, existing_path = find_existing_summary(args.output_dir)
        if not existing_summary.empty:
            print(f"[Info] Existing summary data found: {existing_path}")
            if not args.no_plot:
                plot_density_grouped_bar(existing_summary, args.output_dir, args)
            if args.plot_only or not args.force_rerun:
                print("[Info] Finished plotting from existing data. Use --force-rerun to recompute.")
                return
        elif args.plot_only:
            raise FileNotFoundError(f"No existing summary data found in {args.output_dir}")

    if not args.no_clean_output:
        clean_old_outputs(args.output_dir)

    models = parse_str_list(args.models)
    avg_degrees = parse_float_list(args.avg_degrees)
    raw_records: List[Dict[str, object]] = []

    print("=" * 80)
    print("MSH-only average-degree stress test")
    print("Table and grouped-bar figure output; values summarized as mean ± 95% CI.")
    print(f"N={args.n}, models={models}, target <k>={avg_degrees}, WS p={args.ws_rewiring:g}")
    print(f"instances={args.instances}, repeat-times={args.repeat_times}, master-seed={args.master_seed}")
    print(f"clique-backend={args.clique_backend}")
    print("=" * 80)

    for model in models:
        if model not in {"ER", "BA", "WS"}:
            print(f"[Skip] Unknown model: {model}")
            continue
        for avg_degree in avg_degrees:
            print(f"\n[Condition] model={model}, target <k>={avg_degree:g}")
            for instance in tqdm(range(args.instances), desc=f"{model} k={avg_degree:g}", leave=False):
                g_raw = generate_graph(model, args.n, avg_degree, instance, args.master_seed, args.ws_rewiring)
                raw_n, raw_e = g_raw.number_of_nodes(), g_raw.number_of_edges()
                for repeat in range(args.repeat_times):
                    run_seed = stable_int_hash("run", model, args.n, avg_degree, args.ws_rewiring, instance, repeat, args.master_seed)
                    try:
                        rec = run_full_pipeline_once(g_raw, run_seed, clique_backend=args.clique_backend)
                        rec.update({
                            "Model": model,
                            "Target_Avg_Degree": float(avg_degree),
                            "WS_Rewiring_Probability": float(args.ws_rewiring) if model == "WS" else np.nan,
                            "Instance": int(instance),
                            "Repeat": int(repeat),
                            "Run_Seed": int(run_seed),
                            "Raw_N": int(raw_n),
                            "Raw_E": int(raw_e),
                        })
                        raw_records.append(rec)
                        print(
                            f"  inst={instance}, rep={repeat}: "
                            f"actual_k={rec['Actual_Avg_Degree']:.2f}, C={rec['Average_Clustering']:.4f}, "
                            f"d={rec['Degeneracy']}, #MC={rec['Maximal_Clique_Count']}, maxC={rec['Max_Clique_Size']}, "
                            f"meanMem={rec['Mean_Clique_Membership_Per_Node']:.2f}, maxMem={rec['Max_Clique_Membership_Per_Node']}, "
                            f"pre={rec['Preprocessing_Time_s']:.4f}s, enum={rec['Clique_Enumeration_Time_s']:.4f}s, "
                            f"score={rec['Score_Computation_Time_s']:.4f}s"
                        )
                    except Exception as exc:
                        print(f"  [Error] model={model}, <k>={avg_degree}, instance={instance}, repeat={repeat}: {exc}")
                        traceback.print_exc(limit=3)
            write_outputs(raw_records, args.output_dir, args, checkpoint=True)

    if not raw_records:
        print("No records generated.")
        return

    write_outputs(raw_records, args.output_dir, args, checkpoint=False)
    if not args.no_plot:
        summary_df = summarize(pd.DataFrame(raw_records))
        plot_density_grouped_bar(summary_df, args.output_dir, args)
    print("\n" + "=" * 80)
    print("Done.")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
