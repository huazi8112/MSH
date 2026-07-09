"""
evaluate_cliques_opinion3.py

Purpose
-------
Maximal-clique structure analysis for Reviewer Opinion 3.
The script reports the clique-structure information needed to justify the
clique-aware / mesoscopic interpretation of MSH/HOSH on ordinary pairwise graphs.

Main Opinion-3 indicators
-------------------------
1. Total number of maximal cliques
2. Maximum maximal-clique size
3. Percentage of size-2 maximal cliques
4. Percentage of non-trivial maximal cliques (size >= 3)
5. Node clique participation / membership statistics
6. Graph degeneracy, because Opinion 3 explicitly asks for it and it also
   supports the computational-complexity discussion.

Outputs
-------
results/evaluate_cliques_opinion3/
    Clique_Structure_Opinion3.xlsx
    Clique_Structure_Opinion3_Checkpoint.xlsx
    clique_structure_opinion3_main.csv
    clique_size_distribution.csv
    node_clique_membership_distribution.csv

Run
---
python evaluate_cliques_opinion3.py
python evaluate_cliques_opinion3.py --networks lesmis,jazz,power
"""

from __future__ import annotations

import argparse
import os
import time
import platform
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd

from network_loader import download_and_load_graph, get_network_list


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess_graph(g: nx.Graph) -> nx.Graph:
    """Convert a graph to the simple undirected LCC used by MSH experiments."""
    if g is None:
        return nx.Graph()

    # Convert directed/multigraph graphs to a simple undirected graph.
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        h.add_edges_from((u, v) for u, v in g.edges() if u != v)
    else:
        h = nx.Graph(g.to_undirected()) if g.is_directed() else nx.Graph(g)

    h.remove_edges_from(nx.selfloop_edges(h))

    if h.number_of_nodes() == 0:
        return h

    # Keep the largest connected component for consistency with other experiments.
    if not nx.is_connected(h):
        largest_cc = max(nx.connected_components(h), key=len)
        h = h.subgraph(largest_cc).copy()

    # Relabel nodes to consecutive integers for stable output-independent processing.
    h = nx.convert_node_labels_to_integers(h, ordering="default", label_attribute="original_id")
    return h


# -----------------------------------------------------------------------------
# Clique statistics
# -----------------------------------------------------------------------------
def enumerate_maximal_cliques(g: nx.Graph) -> Tuple[List[List[int]], float]:
    """Enumerate maximal cliques and return elapsed time in seconds."""
    start = time.perf_counter()
    cliques = [list(c) for c in nx.find_cliques(g)]
    elapsed = time.perf_counter() - start
    return cliques, elapsed


def compute_clique_statistics(g: nx.Graph, cliques: List[List[int]], enum_time_s: float) -> Dict[str, Any]:
    """Compute Opinion-3 maximal-clique structure statistics."""
    n = g.number_of_nodes()
    m = g.number_of_edges()

    total_cliques = len(cliques)
    clique_sizes = np.asarray([len(c) for c in cliques], dtype=float)

    if total_cliques > 0:
        size2_count = int(np.sum(clique_sizes == 2))
        size_ge3_count = int(np.sum(clique_sizes >= 3))
        max_clique_size = int(np.max(clique_sizes))
        mean_clique_size = float(np.mean(clique_sizes))
        median_clique_size = float(np.median(clique_sizes))
        pct_size2 = 100.0 * size2_count / total_cliques
        pct_size_ge3 = 100.0 * size_ge3_count / total_cliques
    else:
        size2_count = 0
        size_ge3_count = 0
        max_clique_size = 0
        mean_clique_size = 0.0
        median_clique_size = 0.0
        pct_size2 = 0.0
        pct_size_ge3 = 0.0

    # Node clique memberships: how many maximal cliques each node participates in.
    membership = {node: 0 for node in g.nodes()}
    for c in cliques:
        for node in c:
            membership[node] += 1

    membership_values = np.asarray(list(membership.values()), dtype=float) if membership else np.asarray([], dtype=float)
    if membership_values.size > 0:
        mean_membership = float(np.mean(membership_values))
        median_membership = float(np.median(membership_values))
        max_membership = int(np.max(membership_values))
        pct_nodes_in_ge1 = 100.0 * float(np.sum(membership_values >= 1)) / n if n > 0 else 0.0
        pct_nodes_in_ge2 = 100.0 * float(np.sum(membership_values >= 2)) / n if n > 0 else 0.0
    else:
        mean_membership = 0.0
        median_membership = 0.0
        max_membership = 0
        pct_nodes_in_ge1 = 0.0
        pct_nodes_in_ge2 = 0.0

    degrees = np.asarray([d for _, d in g.degree()], dtype=float)
    avg_degree = float(np.mean(degrees)) if degrees.size > 0 else 0.0
    density = float(nx.density(g)) if n > 1 else 0.0

    try:
        core = nx.core_number(g)
        degeneracy = int(max(core.values())) if core else 0
    except nx.NetworkXError:
        degeneracy = 0

    return {
        "N": int(n),
        "E": int(m),
        "Avg_degree": avg_degree,
        "Density": density,
        "Degeneracy": degeneracy,
        "Total_maximal_cliques": int(total_cliques),
        "Max_clique_size": int(max_clique_size),
        "Mean_clique_size": mean_clique_size,
        "Median_clique_size": median_clique_size,
        "Size2_clique_count": int(size2_count),
        "Pct_size2_cliques": pct_size2,
        "Nontrivial_clique_count_size_ge3": int(size_ge3_count),
        "Pct_nontrivial_cliques_size_ge3": pct_size_ge3,
        "Mean_clique_membership_per_node": mean_membership,
        "Median_clique_membership_per_node": median_membership,
        "Max_clique_membership_per_node": int(max_membership),
        "Pct_nodes_in_at_least_1_clique": pct_nodes_in_ge1,
        "Pct_nodes_in_at_least_2_cliques": pct_nodes_in_ge2,
        "Clique_enumeration_time_s": float(enum_time_s),
    }


def clique_size_distribution(network: str, cliques: List[List[int]]) -> List[Dict[str, Any]]:
    sizes = [len(c) for c in cliques]
    counter = Counter(sizes)
    total = len(sizes)
    rows = []
    for size in sorted(counter):
        count = counter[size]
        rows.append({
            "Network": network,
            "Clique_size": int(size),
            "Count": int(count),
            "Percentage": 100.0 * count / total if total else 0.0,
        })
    return rows


def membership_distribution(network: str, g: nx.Graph, cliques: List[List[int]]) -> List[Dict[str, Any]]:
    membership = {node: 0 for node in g.nodes()}
    for c in cliques:
        for node in c:
            membership[node] += 1

    counter = Counter(membership.values())
    n = g.number_of_nodes()
    rows = []
    for k in sorted(counter):
        count = counter[k]
        rows.append({
            "Network": network,
            "Clique_membership_count": int(k),
            "Node_count": int(count),
            "Percentage": 100.0 * count / n if n else 0.0,
        })
    return rows


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def save_outputs(
    output_dir: str,
    main_records: List[Dict[str, Any]],
    size_rows: List[Dict[str, Any]],
    membership_rows: List[Dict[str, Any]],
    notes: List[Dict[str, str]],
    checkpoint: bool = False,
) -> None:
    main_df = round_numeric_columns(pd.DataFrame(main_records))
    size_df = round_numeric_columns(pd.DataFrame(size_rows))
    membership_df = round_numeric_columns(pd.DataFrame(membership_rows))
    notes_df = pd.DataFrame(notes)

    suffix = "_Checkpoint" if checkpoint else ""
    xlsx_path = os.path.join(output_dir, f"Clique_Structure_Opinion3{suffix}.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="Opinion3_Main", index=False)
        size_df.to_excel(writer, sheet_name="Clique_Size_Distribution", index=False)
        membership_df.to_excel(writer, sheet_name="Node_Membership_Distribution", index=False)
        notes_df.to_excel(writer, sheet_name="Notes", index=False)

    if not checkpoint:
        main_df.to_csv(os.path.join(output_dir, "clique_structure_opinion3_main.csv"), index=False)
        size_df.to_csv(os.path.join(output_dir, "clique_size_distribution.csv"), index=False)
        membership_df.to_csv(os.path.join(output_dir, "node_clique_membership_distribution.csv"), index=False)


def build_notes(args: argparse.Namespace) -> List[Dict[str, str]]:
    return [
        {"Item": "Purpose", "Value": "Maximal-clique structure statistics for Reviewer Opinion 3."},
        {"Item": "Main indicators", "Value": "Total maximal cliques; maximum clique size; size-2 clique ratio; non-trivial clique ratio; node clique participation; degeneracy."},
        {"Item": "Non-trivial clique definition", "Value": "A maximal clique with size >= 3."},
        {"Item": "Preprocessing", "Value": "Simple undirected graph; self-loops removed; largest connected component retained; nodes relabeled to consecutive integers."},
        {"Item": "Clique enumeration", "Value": "networkx.find_cliques, a Bron-Kerbosch-style maximal clique enumeration implementation."},
        {"Item": "Python", "Value": sys.version.replace("\n", " ")},
        {"Item": "Platform", "Value": platform.platform()},
        {"Item": "NetworkX", "Value": nx.__version__},
        {"Item": "NumPy", "Value": np.__version__},
        {"Item": "Pandas", "Value": pd.__version__},
        {"Item": "Command networks", "Value": args.networks if args.networks else "get_network_list()"},
    ]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opinion-3 maximal-clique structure statistics.")
    parser.add_argument(
        "--networks",
        type=str,
        default="",
        help="Comma-separated network names. Default: all networks returned by get_network_list().",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluate_cliques_opinion3",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.networks.strip():
        networks = [x.strip() for x in args.networks.split(",") if x.strip()]
    else:
        networks = list(get_network_list())

    print("=" * 72)
    print("Maximal-clique structure statistics for Reviewer Opinion 3")
    print("=" * 72)
    print(f"Networks: {networks}")
    print(f"Output:   {args.output_dir}")

    main_records: List[Dict[str, Any]] = []
    size_rows: List[Dict[str, Any]] = []
    membership_rows: List[Dict[str, Any]] = []
    notes = build_notes(args)

    for idx, net in enumerate(networks, 1):
        print(f"\n[{idx}/{len(networks)}] Loading network: {net}")
        g_raw = download_and_load_graph(net, verbose=False)
        g = preprocess_graph(g_raw)

        if g.number_of_nodes() == 0:
            print(f"  [Skip] Empty graph after preprocessing: {net}")
            continue

        print(f"  Preprocessed graph: N={g.number_of_nodes()}, E={g.number_of_edges()}")
        cliques, enum_time_s = enumerate_maximal_cliques(g)
        stats = compute_clique_statistics(g, cliques, enum_time_s)

        record = {"Network": net}
        record.update(stats)
        main_records.append(record)
        size_rows.extend(clique_size_distribution(net, cliques))
        membership_rows.extend(membership_distribution(net, g, cliques))

        print(
            "  Cliques={Total_maximal_cliques}, max_size={Max_clique_size}, "
            "size2={Pct_size2_cliques:.2f}%, size>=3={Pct_nontrivial_cliques_size_ge3:.2f}%, "
            "mean_membership={Mean_clique_membership_per_node:.2f}, degeneracy={Degeneracy}".format(**stats)
        )

        # Save checkpoint after every network.
        save_outputs(args.output_dir, main_records, size_rows, membership_rows, notes, checkpoint=True)

    save_outputs(args.output_dir, main_records, size_rows, membership_rows, notes, checkpoint=False)

    main_df = round_numeric_columns(pd.DataFrame(main_records))
    print("\n" + "=" * 72)
    print("Opinion-3 main clique-structure table")
    print("=" * 72)
    if not main_df.empty:
        print(main_df.to_markdown(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
