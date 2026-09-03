#!/usr/bin/env python3
"""Ordered reproduction driver for the second-round MSH repository.

The driver is intentionally thin: it invokes the manuscript-facing scripts in a
fixed order without changing their numerical logic. Use ``--dry-run`` or
``--list`` before a full execution. The complete run is computationally
expensive because it includes SIR and synthetic runtime experiments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    command: List[str]


PYTHON = sys.executable
STEPS = [
    Step("validate", "Repository validation", [PYTHON, "-m", "tools.validate_repository"]),
    Step("references", "Reference-table regression validation", [PYTHON, "-m", "tools.validate_reference_results"]),
    Step("tableS1", "Preprocessing comparison (Table S1)", [PYTHON, "-m", "tools.evaluate_preprocessing"]),
    Step("table1", "Processed network statistics (Table 1)", [PYTHON, "-m", "tools.compute_network_statistics"]),
    Step("table2", "Clique-structure statistics (Table 2)", [PYTHON, "-m", "tools.evaluate_clique_structure"]),
    Step("rankings", "Precompute rankings and clique caches", [PYTHON, "precompute_rankings.py", "--force"]),
    Step("fig3", "Seed-ratio SIR (Fig. 3)", [PYTHON, "-m", "experiments.exp01_seed_ratio_sir_fig3"]),
    Step("tableS4", "Cross-network aggregate statistics (Table S4)", [PYTHON, "-m", "experiments.exp02_cross_network_stats_tableS4"]),
    Step("fig4", "Temporal SIR + Table 3", [PYTHON, "-m", "experiments.exp03_temporal_sir_fig4_table3"]),
    Step("fig5", "Beta robustness (Fig. 5)", [PYTHON, "-m", "experiments.exp04_beta_robustness_fig5"]),
    Step("table4", "Monotonicity (Table 4)", [PYTHON, "-m", "experiments.exp05_monotonicity_table4"]),
    Step("fig6", "Ranking-frequency distributions (Fig. 6)", [PYTHON, "-m", "experiments.exp06_ranking_frequency_fig6"]),
    Step("fig7", "Topological dispersion (Fig. 7)", [PYTHON, "-m", "experiments.exp07_topological_dispersion_fig7"]),
    Step("fig8", "Topology visualization (Fig. 8)", [PYTHON, "-m", "experiments.exp08_topology_visualization_fig8"]),
    Step("fig9", "Structural similarity (Fig. 9)", [PYTHON, "-m", "experiments.exp09_structural_similarity_fig9"]),
    Step("fig10", "Matched controls (Fig. 10, Tables S6-S7)", [PYTHON, "-m", "experiments.exp10_matched_controls_fig10_tablesS6S7"]),
    Step("fig11data", "All-node Kendall correlations", [PYTHON, "-m", "experiments.exp11_kendall_correlation_fig11"]),
    Step("fig11", "Plot Kendall correlation matrix (Fig. 11)", [PYTHON, "-m", "experiments.plot_fig11_kendall"]),
    Step("table5", "Ablation analysis (Table 5, Tables S8-S9)", [PYTHON, "-m", "experiments.exp12_ablation_table5_tablesS8S9"]),
    Step("fig12", "Network-size runtime comparison (Fig. 12)", [PYTHON, "-m", "experiments.exp13_runtime_scale_fig12"]),
    Step("fig13", "Average-degree stress (Fig. 13, Table S10)", [PYTHON, "-m", "experiments.exp14_average_degree_stress_fig13_tableS10"]),
    Step("tableS2", "Tie-breaking sensitivity (Table S2)", [PYTHON, "-m", "experiments.exp15_tie_breaking_tableS2"]),
    Step("tableS5", "Recovery-rate robustness (Table S5)", [PYTHON, "-m", "experiments.exp16_recovery_rate_tableS5"]),
    Step("tableS11", "Clustering stress (Table S11)", [PYTHON, "-m", "experiments.exp17_clustering_stress_tableS11"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ordered MSH second-round reproduction workflow.")
    parser.add_argument("--list", action="store_true", help="List step keys and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the listed step keys, in canonical order.")
    parser.add_argument("--from-step", default=None, help="Start from this step key.")
    parser.add_argument("--through-step", default=None, help="Stop after this step key.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed step.")
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> List[Step]:
    keys = [s.key for s in STEPS]
    chosen = STEPS
    if args.from_step:
        if args.from_step not in keys:
            raise SystemExit(f"Unknown --from-step {args.from_step!r}. Use --list.")
        chosen = chosen[keys.index(args.from_step):]
    if args.through_step:
        if args.through_step not in keys:
            raise SystemExit(f"Unknown --through-step {args.through_step!r}. Use --list.")
        stop = keys.index(args.through_step)
        chosen = [s for s in chosen if keys.index(s.key) <= stop]
    if args.only is not None:
        unknown = [k for k in args.only if k not in keys]
        if unknown:
            raise SystemExit(f"Unknown --only keys: {unknown}. Use --list.")
        wanted = set(args.only)
        chosen = [s for s in chosen if s.key in wanted]
    return chosen


def main() -> int:
    args = parse_args()
    if args.list:
        for i, step in enumerate(STEPS, start=1):
            print(f"{i:02d}  {step.key:<12} {step.label}")
        return 0

    chosen = selected_steps(args)
    if not chosen:
        print("No steps selected.")
        return 0

    failures = []
    for i, step in enumerate(chosen, start=1):
        cmd_text = " ".join(step.command)
        print(f"\n[{i}/{len(chosen)}] {step.key}: {step.label}")
        print(f"$ {cmd_text}")
        if args.dry_run:
            continue
        proc = subprocess.run(step.command, cwd=ROOT)
        if proc.returncode != 0:
            failures.append((step.key, proc.returncode))
            print(f"[FAILED] {step.key} returned {proc.returncode}")
            if not args.continue_on_error:
                return proc.returncode
        else:
            print(f"[OK] {step.key}")

    if failures:
        print("\nCompleted with failures:")
        for key, code in failures:
            print(f"- {key}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
