"""Audit the explicit VoteRank ascending-node-ID internal tie rule.

This migration audit compares the uploaded revision implementation (which relied
on candidate-set iteration order when voting scores tied) with the final explicit
rule (smallest node ID wins a voting-score tie). It is not part of the manuscript
inference; it documents that the reproducibility cleanup does not change the
reported VoteRank ordering on the nine empirical graphs.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from msh_methods import calculate_voterank
from network_loader import download_and_load_graph, get_network_list

ROOT = Path(__file__).resolve().parents[1]


def legacy_voterank(g):
    N = g.number_of_nodes()
    scores = {n: 0.0 for n in g.nodes()}
    if N == 0:
        return scores, 0
    avg_degree = sum(dict(g.degree()).values()) / N
    f = 1.0 / avg_degree if avg_degree > 0 else 0.0
    voting_ability = {n: 1.0 for n in g.nodes()}
    candidates = set(g.nodes())
    neighbors_map = {n: list(g.neighbors(n)) for n in g.nodes()}
    rank = 0
    tie_steps = 0
    while candidates:
        best_node = None
        max_score = -1.0
        tied = []
        for node in candidates:
            current_score = sum(voting_ability[neighbor] for neighbor in neighbors_map[node])
            if current_score > max_score:
                max_score = current_score
                best_node = node
                tied = [node]
            elif current_score == max_score:
                tied.append(node)
        if len(tied) > 1:
            tie_steps += 1
        scores[best_node] = float(N - rank)
        rank += 1
        candidates.remove(best_node)
        voting_ability[best_node] = 0.0
        for neighbor in neighbors_map[best_node]:
            if voting_ability[neighbor] > 0:
                voting_ability[neighbor] = max(0.0, voting_ability[neighbor] - f)
    return scores, tie_steps


def ordered(scores):
    return [n for n, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def main() -> int:
    rows = []
    for network in get_network_list():
        g = download_and_load_graph(network, verbose=False)
        legacy_scores, tie_steps = legacy_voterank(g)
        final_scores = calculate_voterank(g)
        legacy_order = ordered(legacy_scores)
        final_order = ordered(final_scores)
        N = g.number_of_nodes()
        row = {
            "Network": network,
            "N": N,
            "Legacy_tied_selection_steps": tie_steps,
            "Full_order_identical": legacy_order == final_order,
            "Top10_set_identical": set(legacy_order[:10]) == set(final_order[:10]),
        }
        for p in range(1, 11):
            k = max(1, int(N * p / 100.0))
            row[f"Top{p}pct_set_difference_count"] = len(set(legacy_order[:k]) ^ set(final_order[:k])) // 2
        rows.append(row)

    df = pd.DataFrame(rows)
    out = ROOT / "docs" / "voterank_tie_audit.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")
    return 0 if bool(df["Full_order_identical"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
