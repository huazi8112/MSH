"""
Tie-breaking sensitivity experiment: final CV across ID, Random, and SIR_GT strategies.

Purpose
-------
Evaluate whether ties in node rankings materially affect the final SIR spreading
performance for the main ranking methods only. Ablation variants are excluded.
The official protocol uses deterministic node-ID tie-breaking.
Random and SIR_GT tie-breaking are reported only as sensitivity checks.

Tie-breaking strategies
-----------------------
1. ID      : official deterministic rule. If the Top-K boundary cuts through a
             group of equal-score nodes, the remaining slots are filled by
             ascending node ID after graph relabeling.
2. Random  : sensitivity rule. Only the equal-score boundary group is shuffled
             using deterministic pseudo-random seeds derived from blake2b.
3. SIR_GT  : sensitivity/oracle-like rule. Only the equal-score boundary group
             is sorted by single-node SIR estimates; node ID is used as a
             secondary deterministic tie-breaker.

Main output metric
------------------
For each network-method-seed-ratio setting, the script reports:

    CV_final(%) = std([F_ID, F_Random, F_SIR_GT])
                  / mean([F_ID, F_Random, F_SIR_GT]) * 100

where F_ID, F_Random, and F_SIR_GT are the mean final infection scales under the
three tie-breaking strategies. F_Random is averaged over the requested number of
random boundary-tie trials.

Reproducibility
---------------
- No Python built-in hash() is used.
- All random seeds are derived by stable blake2b hashing.
- SIR simulations use local random.Random instances.
- The same SIR random-seed blocks are shared across tie-breaking strategies for
  each network/method/seed-ratio setting.

Outputs
-------
results/exp_tie_breaking_cv_three_strategies_simple_table/
  - tie_breaking_final_cv_summary.xlsx
  - tie_breaking_cv_manuscript_table.csv
  - tie_breaking_final_cv_summary.csv
  - tie_breaking_strategy_long.csv
  - tie_breaking_checkpoint.xlsx

Manuscript table
----------------
The main table contains only four columns:
    Method | p=1% Mean CV | p=5% Mean CV | p=10% Mean CV
No overall mean, max CV, or confidence interval columns are included.
"""

import argparse
import hashlib
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from precompute_rankings import load_precomputed_rankings
from network_loader import download_and_load_graph, get_network_list


MASTER_SEED = 42

# Main methods only. Ablation variants such as HOSH-NO, HOSH-NE, HOSH-E,
# HOSH-C, HOSH-Lin, HOSH-BoxCox, and HOSH-SumNorm are intentionally excluded
# from this tie-breaking sensitivity experiment.
DEFAULT_METHODS = [
    "HOSH", "VoteRank", "SNIM", "CHBC",
    "ISH", "DC", "BC", "CC", "K-Shell", "SH", "CI", "SNC",
]

METHOD_LABELS = {
    "HOSH": "MSH",
}


# =========================================================
# 0. Reproducibility utilities
# =========================================================
def stable_int_hash(*items, modulo: int = 2**32 - 1) -> int:
    """Stable integer hash for deterministic seed derivation across processes."""
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def stable_text_digest(items: Iterable, digest_size: int = 8) -> str:
    """Stable short digest for recording seed-set identity without storing long lists."""
    text = "||".join(str(x) for x in items)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def set_master_seed(seed: int = MASTER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_node_sort_key(n):
    """Robust deterministic node ordering for numeric or string node IDs."""
    try:
        return (0, int(n))
    except Exception:
        return (1, str(n))


def seed_set_digest(seeds: Sequence) -> str:
    ordered = sorted(seeds, key=safe_node_sort_key)
    return stable_text_digest(ordered)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# =========================================================
# 1. SIR model and evaluation
# =========================================================
def epidemic_parameters(graph: nx.Graph, gamma: float, multiplier: float, cap_beta: bool = True):
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    k_mean = float(np.mean(degrees)) if len(degrees) else 0.0
    k2_mean = float(np.mean(degrees ** 2)) if len(degrees) else 0.0
    denom = k2_mean - k_mean
    beta_th = k_mean / denom if denom > 0 else 0.0
    beta = multiplier * beta_th
    if cap_beta:
        beta = min(beta, 1.0)
    return beta_th, beta, gamma


def run_sir_simulation(
    graph: nx.Graph,
    seeds: Iterable,
    beta: float,
    gamma: float,
    rng_seed: int,
    max_steps: int = 1000,
) -> int:
    """Discrete-time synchronous SIR simulation using a local RNG."""
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
            # Infection attempts are evaluated before recovery.
            for v in graph.neighbors(u):
                if v not in infected and v not in recovered:
                    if rng.random() < beta:
                        new_infected.add(v)

            # Recovery is evaluated in the same synchronous time step.
            if rng.random() < gamma:
                new_recovered.add(u)

        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)

    return len(recovered) + len(infected)


def single_node_sir_score(
    graph: nx.Graph,
    node,
    beta: float,
    gamma: float,
    network_name: str,
    method: str,
    ratio_pct: int,
    repeats: int,
    base_seed: int = MASTER_SEED,
) -> float:
    """Single-node SIR estimate used only for SIR_GT boundary tie-breaking."""
    vals = []
    for r in range(repeats):
        seed = stable_int_hash(base_seed, "single_node_sir", network_name, method, ratio_pct, node, r)
        vals.append(run_sir_simulation(graph, [node], beta, gamma, rng_seed=seed))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_seed_set_blocks(
    graph: nx.Graph,
    seeds: Sequence,
    beta: float,
    gamma: float,
    network_name: str,
    method: str,
    ratio_pct: int,
    blocks: int,
    repeats: int,
    base_seed: int = MASTER_SEED,
) -> np.ndarray:
    """
    Return block-level final infection scales in percent.
    SIR random seeds are independent of tie-breaking strategy, enabling paired comparison.
    """
    n = graph.number_of_nodes()
    block_values = []

    for b in range(blocks):
        vals = []
        for r in range(repeats):
            seed = stable_int_hash(base_seed, "tie_eval", network_name, method, ratio_pct, b, r)
            impact = run_sir_simulation(graph, seeds, beta, gamma, rng_seed=seed)
            vals.append(impact / n * 100.0)
        block_values.append(float(np.mean(vals)))

    return np.asarray(block_values, dtype=float)


def summarize_blocks(block_values: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean, standard deviation, and 95% CI half-width from block-level values."""
    arr = np.asarray(block_values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean_val, std_val, ci95


# =========================================================
# 2. Tie-breaking utilities
# =========================================================
def grouped_scores(scores: Dict, round_decimals: int = 6) -> Dict[float, List]:
    """Group nodes by rounded scores to avoid meaningless floating-point distinctions."""
    groups = defaultdict(list)
    for node, score in scores.items():
        try:
            val = float(score)
            if np.isfinite(val):
                groups[round(val, round_decimals)].append(node)
        except Exception:
            continue
    return groups


def inspect_boundary_tie(scores: Dict, graph: nx.Graph, top_k: int, round_decimals: int = 6) -> Dict:
    """Inspect whether Top-K selection cuts through an equal-score boundary group."""
    groups = grouped_scores(scores, round_decimals=round_decimals)
    sorted_score_values = sorted(groups.keys(), reverse=True)
    selected_count = 0

    for score_value in sorted_score_values:
        group = [n for n in groups[score_value] if graph.has_node(n)]
        group = sorted(group, key=safe_node_sort_key)
        need = top_k - selected_count
        if need <= 0:
            break
        if len(group) <= need:
            selected_count += len(group)
            continue

        return {
            "Boundary_Tie": "Yes",
            "Boundary_Score": score_value,
            "Boundary_Group_Size": len(group),
            "Boundary_Need": need,
            "Boundary_Unselected": len(group) - need,
        }

    return {
        "Boundary_Tie": "No",
        "Boundary_Score": np.nan,
        "Boundary_Group_Size": 0,
        "Boundary_Need": 0,
        "Boundary_Unselected": 0,
    }


def get_top_k_with_strategy(
    scores: Dict,
    graph: nx.Graph,
    top_k: int,
    beta: float,
    gamma: float,
    strategy: str,
    network_name: str,
    method: str,
    ratio_pct: int,
    single_node_repeats: int,
    random_trial: int = 0,
    round_decimals: int = 6,
) -> List:
    """
    Select Top-K nodes. Only the boundary tie group is resolved by the chosen strategy.
    """
    groups = grouped_scores(scores, round_decimals=round_decimals)
    sorted_score_values = sorted(groups.keys(), reverse=True)
    selected = []

    for score_value in sorted_score_values:
        group = [n for n in groups[score_value] if graph.has_node(n)]
        group = sorted(group, key=safe_node_sort_key)
        need = top_k - len(selected)

        if need <= 0:
            break

        if len(group) <= need:
            selected.extend(group)
            continue

        # The Top-K cutoff falls inside this equal-score group.
        if strategy == "id":
            ordered = sorted(group, key=safe_node_sort_key)
        elif strategy == "random":
            ordered = list(group)
            rng = random.Random(
                stable_int_hash(MASTER_SEED, "boundary_random", network_name, method, ratio_pct, random_trial)
            )
            rng.shuffle(ordered)
        elif strategy == "sir_gt":
            ordered = sorted(
                group,
                key=lambda n: (
                    -single_node_sir_score(
                        graph,
                        n,
                        beta,
                        gamma,
                        network_name,
                        method,
                        ratio_pct,
                        repeats=single_node_repeats,
                        base_seed=MASTER_SEED,
                    ),
                    safe_node_sort_key(n),
                ),
            )
        else:
            raise ValueError(f"Unknown tie-breaking strategy: {strategy}")

        selected.extend(ordered[:need])
        break

    return selected[:top_k]


def coefficient_of_variation(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    mean_val = float(np.mean(values))
    if mean_val <= 0:
        return 0.0
    return float(np.std(values, ddof=0) / mean_val * 100.0)


# =========================================================
# 3. Argument parsing
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tie-breaking sensitivity: CV across ID, Random, and SIR_GT final infection scales."
    )
    parser.add_argument("--networks", type=str, default="", help="Comma-separated networks. Default: all networks.")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated main methods. Ablation variants are excluded by default.",
    )
    parser.add_argument("--seed-ratios", type=str, default="1,5,10", help="Seed ratios in percent, e.g., 1,5,10.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Recovery probability. Default: 1.0.")
    parser.add_argument("--threshold-multiplier", type=float, default=2.5, help="beta = lambda * beta_th. Default: 2.5.")
    parser.add_argument("--no-cap-beta", action="store_true", help="Do not cap beta at 1.0.")
    parser.add_argument("--blocks", type=int, default=50, help="Independent random-seed blocks for SIR evaluation.")
    parser.add_argument("--repeats", type=int, default=20, help="SIR repetitions per block.")
    parser.add_argument(
        "--single-node-repeats",
        type=int,
        default=20,
        help="Repeats for SIR_GT scoring within the boundary tie group only.",
    )
    parser.add_argument(
        "--random-tie-trials",
        type=int,
        default=5,
        help="Number of deterministic random boundary-tie trials used to estimate the Random strategy.",
    )
    parser.add_argument("--round-decimals", type=int, default=6, help="Score rounding decimals for tie detection.")
    parser.add_argument("--output-dir", type=str, default="results/exp_tie_breaking_cv_three_strategies_simple_table")
    return parser.parse_args()


# =========================================================
# 4. Main workflow
# =========================================================
def main():
    args = parse_args()
    set_master_seed(MASTER_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    networks = [x.strip() for x in args.networks.split(",") if x.strip()] if args.networks.strip() else get_network_list()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()] if args.methods.strip() else list(DEFAULT_METHODS)
    # Remove ablation variants unless the user explicitly passes them through --methods.
    methods = [m for m in methods if m in DEFAULT_METHODS or args.methods.strip()]
    ratio_pcts = [int(float(x.strip())) for x in args.seed_ratios.split(",") if x.strip()]

    print("=" * 76)
    print("Tie-breaking sensitivity: final CV across ID, Random, and SIR_GT")
    print("Official strategy: ID. Random and SIR_GT are sensitivity checks only.")
    print(f"Networks: {networks}")
    print(f"Methods: {methods}")
    print(f"Seed ratios: {ratio_pcts}%")
    print(
        f"SIR: gamma={args.gamma}, lambda={args.threshold_multiplier}, "
        f"blocks={args.blocks}, repeats={args.repeats}, random_tie_trials={args.random_tie_trials}"
    )
    print("=" * 76)

    strategy_definitions = pd.DataFrame(
        [
            {
                "Strategy": "ID",
                "Role": "Official rule",
                "Definition": "When the Top-K boundary cuts through equal-score nodes, select by ascending node ID after relabeling.",
            },
            {
                "Strategy": "Random",
                "Role": "Sensitivity check",
                "Definition": "Randomly shuffle only the equal-score boundary group using deterministic blake2b-derived seeds.",
            },
            {
                "Strategy": "SIR_GT",
                "Role": "Sensitivity/oracle-like check",
                "Definition": "Rank only the equal-score boundary group by single-node SIR estimates; use node ID as secondary tie-breaker.",
            },
        ]
    )

    summary_records = []
    strategy_records = []

    for net_idx, net in enumerate(networks, 1):
        print(f"\n[{net_idx}/{len(networks)}] Network: {net}")
        graph = download_and_load_graph(net, verbose=False)
        if graph is None or graph.number_of_nodes() == 0:
            print(f"  [Skip] {net}: failed to load or empty graph.")
            continue

        n = graph.number_of_nodes()
        beta_th, beta, gamma = epidemic_parameters(
            graph, args.gamma, args.threshold_multiplier, cap_beta=(not args.no_cap_beta)
        )
        print(f"  Nodes={n}, Edges={graph.number_of_edges()}, beta_th={beta_th:.6f}, beta={beta:.6f}")

        precomputed = load_precomputed_rankings(net)
        if not precomputed:
            print(f"  [Skip] {net}: no precomputed rankings found.")
            continue

        for method in tqdm(methods, desc=f"  Methods for {net}"):
            if method not in precomputed or precomputed[method] is None:
                continue
            scores = precomputed[method]

            for ratio_pct in ratio_pcts:
                top_k = max(1, int(np.ceil(n * ratio_pct / 100.0)))
                boundary_info = inspect_boundary_tie(scores, graph, top_k, round_decimals=args.round_decimals)

                # Official deterministic ID rule.
                seeds_id = get_top_k_with_strategy(
                    scores,
                    graph,
                    top_k,
                    beta,
                    gamma,
                    "id",
                    net,
                    method,
                    ratio_pct,
                    single_node_repeats=args.single_node_repeats,
                    round_decimals=args.round_decimals,
                )
                blocks_id = evaluate_seed_set_blocks(
                    graph, seeds_id, beta, gamma, net, method, ratio_pct, blocks=args.blocks, repeats=args.repeats
                )
                f_id, std_id, ci_id = summarize_blocks(blocks_id)

                if boundary_info["Boundary_Tie"] == "No":
                    # If the Top-K boundary does not cut through an equal-score group,
                    # all three strategies select exactly the same seed set. Avoid
                    # unnecessary SIR_GT and random-tie computations.
                    seeds_sir = list(seeds_id)
                    f_sir, std_sir, ci_sir = f_id, std_id, ci_id
                    f_random, std_random, ci_random = f_id, std_id, ci_id
                    random_selection_std = 0.0
                    random_unique_seed_sets = 1
                    random_seed_digests = [seed_set_digest(seeds_id)]
                else:
                    # Oracle-like SIR_GT boundary rule.
                    seeds_sir = get_top_k_with_strategy(
                        scores,
                        graph,
                        top_k,
                        beta,
                        gamma,
                        "sir_gt",
                        net,
                        method,
                        ratio_pct,
                        single_node_repeats=args.single_node_repeats,
                        round_decimals=args.round_decimals,
                    )
                    blocks_sir = evaluate_seed_set_blocks(
                        graph, seeds_sir, beta, gamma, net, method, ratio_pct, blocks=args.blocks, repeats=args.repeats
                    )
                    f_sir, std_sir, ci_sir = summarize_blocks(blocks_sir)

                    # Random boundary rule. Average over deterministic random boundary-tie trials.
                    random_trial_means = []
                    random_block_matrix = []
                    random_seed_digests = []

                    for trial in range(args.random_tie_trials):
                        seeds_rand = get_top_k_with_strategy(
                            scores,
                            graph,
                            top_k,
                            beta,
                            gamma,
                            "random",
                            net,
                            method,
                            ratio_pct,
                            single_node_repeats=args.single_node_repeats,
                            random_trial=trial,
                            round_decimals=args.round_decimals,
                        )
                        blocks_rand = evaluate_seed_set_blocks(
                            graph, seeds_rand, beta, gamma, net, method, ratio_pct, blocks=args.blocks, repeats=args.repeats
                        )
                        random_block_matrix.append(blocks_rand)
                        random_trial_means.append(float(np.mean(blocks_rand)))
                        random_seed_digests.append(seed_set_digest(seeds_rand))

                    if random_block_matrix:
                        random_block_matrix = np.vstack(random_block_matrix)
                        random_blocks_mean_by_block = np.mean(random_block_matrix, axis=0)
                        f_random, std_random, ci_random = summarize_blocks(random_blocks_mean_by_block)
                        random_selection_std = float(np.std(random_trial_means, ddof=1)) if len(random_trial_means) > 1 else 0.0
                        random_unique_seed_sets = len(set(random_seed_digests))
                    else:
                        f_random, std_random, ci_random = np.nan, np.nan, np.nan
                        random_selection_std = np.nan
                        random_unique_seed_sets = 0

                final_values = np.asarray([f_id, f_random, f_sir], dtype=float)
                cv_final = coefficient_of_variation(final_values)
                final_mean = float(np.nanmean(final_values))
                max_diff = float(np.nanmax(final_values) - np.nanmin(final_values))
                relative_range = (max_diff / final_mean * 100.0) if final_mean > 0 else 0.0

                base_record = {
                    "Network": net,
                    "Method": method,
                    "MethodLabel": method_label(method),
                    "p(%)": ratio_pct,
                    "k": top_k,
                    "gamma": args.gamma,
                    "lambda": args.threshold_multiplier,
                    "beta_th": round(beta_th, 8),
                    "beta": round(beta, 8),
                    "Blocks": args.blocks,
                    "Repeats": args.repeats,
                    "Single_Node_Repeats": args.single_node_repeats,
                    "Random_Tie_Trials": args.random_tie_trials,
                    "Master_Seed": MASTER_SEED,
                    **boundary_info,
                }

                summary_records.append(
                    {
                        **base_record,
                        "F_ID(%)": round(f_id, 4),
                        "F_Random(%)": round(f_random, 4),
                        "F_SIR_GT(%)": round(f_sir, 4),
                        "CV_final(%)": round(cv_final, 6),
                        "Mean_of_3_F(%)": round(final_mean, 4),
                        "Max_Diff_among_3(%)": round(max_diff, 4),
                        "Relative_Range_among_3(%)": round(relative_range, 6),
                        "Random_Selection_STD(%)": round(random_selection_std, 6),
                        "Random_Unique_Seed_Sets": random_unique_seed_sets,
                        "SeedSetHash_ID": seed_set_digest(seeds_id),
                        "SeedSetHash_SIR_GT": seed_set_digest(seeds_sir),
                    }
                )

                for strategy_name, f_val, std_val, ci_val, digest in [
                    ("ID", f_id, std_id, ci_id, seed_set_digest(seeds_id)),
                    ("Random", f_random, std_random, ci_random, "multiple" if args.random_tie_trials > 1 else (random_seed_digests[0] if random_seed_digests else "NA")),
                    ("SIR_GT", f_sir, std_sir, ci_sir, seed_set_digest(seeds_sir)),
                ]:
                    strategy_records.append(
                        {
                            **base_record,
                            "Strategy": strategy_name,
                            "F_mean(%)": round(f_val, 4),
                            "F_block_STD(%)": round(std_val, 6),
                            "F_CI95_half_width(%)": round(ci_val, 6),
                            "SeedSetHash": digest,
                        }
                    )

        # Checkpoint after each network.
        if summary_records:
            checkpoint_path = os.path.join(args.output_dir, "tie_breaking_checkpoint.xlsx")
            with pd.ExcelWriter(checkpoint_path, engine="openpyxl") as writer:
                pd.DataFrame(summary_records).to_excel(writer, sheet_name="Final_CV", index=False)
                pd.DataFrame(strategy_records).to_excel(writer, sheet_name="Strategy_Long", index=False)
                strategy_definitions.to_excel(writer, sheet_name="Strategy_Definition", index=False)

    if not summary_records:
        print("No valid records were generated.")
        return

    summary_df = pd.DataFrame(summary_records)
    strategy_df = pd.DataFrame(strategy_records)

    # Compact manuscript table: mean CV across networks for each method and seed ratio.
    # Only p-specific mean CV columns are reported; no overall mean, max CV, or CI columns.
    manuscript_long = (
        summary_df.groupby(["MethodLabel", "p(%)"], dropna=False)["CV_final(%)"]
        .mean()
        .reset_index()
        .rename(columns={"MethodLabel": "Method", "CV_final(%)": "Mean_CV(%)"})
    )
    manuscript_table = manuscript_long.pivot(index="Method", columns="p(%)", values="Mean_CV(%)").reset_index()

    # Preserve the default method order and rename seed-ratio columns.
    ordered_labels = [method_label(m) for m in methods]
    manuscript_table["__order"] = manuscript_table["Method"].map(
        {label: idx for idx, label in enumerate(ordered_labels)}
    )
    manuscript_table = manuscript_table.sort_values("__order").drop(columns="__order")
    rename_cols = {p: f"p={int(p)}% Mean CV" for p in ratio_pcts if p in manuscript_table.columns}
    manuscript_table = manuscript_table.rename(columns=rename_cols)
    for col in manuscript_table.columns:
        if col != "Method":
            manuscript_table[col] = manuscript_table[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)

    # Detail table for appendix or checking. This keeps every network-method-ratio CV.
    detail_cv_table = summary_df[[
        "Network", "MethodLabel", "p(%)", "k", "Boundary_Tie", "Boundary_Group_Size",
        "F_ID(%)", "F_Random(%)", "F_SIR_GT(%)", "CV_final(%)"
    ]].copy().rename(columns={"MethodLabel": "Method"})

    csv_summary_path = os.path.join(args.output_dir, "tie_breaking_final_cv_summary.csv")
    csv_strategy_path = os.path.join(args.output_dir, "tie_breaking_strategy_long.csv")
    csv_manuscript_path = os.path.join(args.output_dir, "tie_breaking_cv_manuscript_table.csv")
    xlsx_path = os.path.join(args.output_dir, "tie_breaking_final_cv_summary.xlsx")

    summary_df.to_csv(csv_summary_path, index=False, encoding="utf-8-sig")
    strategy_df.to_csv(csv_strategy_path, index=False, encoding="utf-8-sig")
    manuscript_table.to_csv(csv_manuscript_path, index=False, encoding="utf-8-sig")
    detail_cv_table.to_csv(os.path.join(args.output_dir, "tie_breaking_cv_detail_table.csv"), index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        manuscript_table.to_excel(writer, sheet_name="Manuscript_Table", index=False)
        detail_cv_table.to_excel(writer, sheet_name="Detail_CV", index=False)
        summary_df.to_excel(writer, sheet_name="Final_CV", index=False)
        strategy_df.to_excel(writer, sheet_name="Strategy_Long", index=False)
        strategy_definitions.to_excel(writer, sheet_name="Strategy_Definition", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False

    print("\n" + "=" * 76)
    print("Done.")
    print(f"Manuscript table saved: {csv_manuscript_path}")
    print(f"Summary CSV saved: {csv_summary_path}")
    print(f"Strategy CSV saved: {csv_strategy_path}")
    print(f"Excel saved: {xlsx_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
