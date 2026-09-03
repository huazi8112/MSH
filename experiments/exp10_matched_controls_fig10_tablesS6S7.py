"""
Opinion 10 experiment: dispersion-control quadrant scatter plots on nine networks.

Purpose
-------
This script is independent of previous experimental result files. It recomputes:
  1) MSH/HOSH seed sets;
  2) degree-matched random controls (DMR);
  3) degree-matched dispersed controls (DMD);
  4) degree-and-distance-matched random controls (DDMR) with an explicit L_s caliper;
  5) final SIR infection scale F(tc);
  6) seed-set spatial dispersion L_s.

It then draws quadrant scatter plots for each of the nine networks.
Each scatter point corresponds to one seed ratio p = 1%, ..., 10% and one control.

Coordinates
-----------
For each control method C in {DMR, DMD, DDMR}, define:
    x = Delta L_s = L_s(C) - L_s(MSH)
    y = Delta F(t_c) = F_MSH(t_c) - F_C(t_c)

Interpretation
--------------
- y > 0 means MSH has a larger final infection scale than the control at t_c.
- x > 0 means the control is more spatially dispersed than MSH.
- DDMR is the most important control when many DMD points fall in the second quadrant,
  because it explicitly controls both degree profile and spatial dispersion.
- In this revised version, DDMR is selected only from candidates whose relative L_s
  error from MSH is within the specified caliper, 5% by default.

Outputs
-------
results/exp_opinion10_ddmr_quadrant_9networks/
  - Opinion10_DDMR_Quadrant_9Networks.pdf / .png
  - Opinion10_DDMR_Quadrant_<network>.pdf / .png
  - Opinion10_DDMR_Quadrant_Legend_Standalone.pdf / .png
  - Opinion10_DDMR_Quadrant_Data.xlsx
  - Opinion10_DDMR_Quadrant_Checkpoint.xlsx

Default parameters
------------------
  p = 1%, 2%, ..., 10%
  gamma = 1.0
  beta = 2.5 * beta_th
  beta_th = <k> / (<k^2> - <k>) by default
  50 blocks x 20 repeats for SIR evaluation
  DMR = 10 random degree-matched sets
  DMD = 1 maximum-dispersion set from the 1000-candidate pool
  DDMR = target 10 random valid sets within a 5% L_s caliper; if only 1-9 valid sets exist after the maximum search, all are used

Run
---
  python exp_opinion10_ddmr_quadrant_9networks_caliper5_abbrev_consistent_labels.py

Quick test
----------
  python exp_opinion10_ddmr_quadrant_9networks_caliper5_abbrev_consistent_labels.py --networks jazz,email,usair --blocks 5 --repeats 5 --control-trials 200 --ddmr-max-candidates 2000 --ddmr-nearest 5

If the manuscript uses the name MSH while the implementation still uses HOSH:
  python exp_opinion10_ddmr_quadrant_9networks_caliper5_abbrev_consistent_labels.py --proposed-method HOSH --proposed-label MSH
"""

import argparse
import math
import os
import random
import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t, wilcoxon, rankdata
from tqdm import tqdm

from hosh_methods import get_node_scores
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings


# =========================================================
# 0. Plot configuration: SIR-style, standalone legend, PDFs
# =========================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5.0,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})

CONTROL_STYLES = {
    'DMR': {
        'label': 'DMR',
        'color': '#4FA3D1',
        'marker': 'o',
    },
    'DMD': {
        'label': 'DMD',
        'color': '#F08C3D',
        'marker': '^',
    },
    'DDMR': {
        'label': 'DDMR',
        'color': '#5D3FD3',
        'marker': 's',
    },
}


NETWORK_DISPLAY_NAMES = {
    'jazz': 'Jazz',
    'email': 'Email',
    'hamster': 'Hamster',
    'usair': 'USAir',
    'polblogs': 'PolBlogs',
    'power': 'Power',
    'lesmis': 'Lesmis',
    'infect': 'Infect',
    'netsci': 'NetSci',
}


def format_network_name(name):
    """Return a publication-style network name for figure titles."""
    key = str(name).strip().lower()
    return NETWORK_DISPLAY_NAMES.get(key, str(name).strip().capitalize())


# =========================================================
# 1. General utilities
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(obj, modulo=1000000):
    """Deterministic hash independent of Python's randomized hash seed.

    Python's built-in hash() is intentionally randomized across interpreter
    processes unless PYTHONHASHSEED is fixed. This helper keeps all derived
    random seeds reproducible across repeated script runs.
    """
    s = str(obj)
    h = 2166136261  # FNV-1a 32-bit offset basis
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xffffffff
    return h % modulo


def safe_node_sort_key(n):
    """Robust deterministic sort key for integer or string node IDs."""
    try:
        return (0, int(n))
    except Exception:
        return (1, str(n))


def rank_nodes_from_scores(scores: dict):
    """Sort nodes by descending score; ties are broken deterministically by node ID."""
    clean_items = []
    for n, s in scores.items():
        try:
            val = float(s)
            if np.isfinite(val):
                clean_items.append((n, val))
        except Exception:
            continue
    clean_items.sort(key=lambda x: (-x[1], safe_node_sort_key(x[0])))
    return [n for n, _ in clean_items]


def get_proposed_ranking(g, network_name, proposed_method='HOSH', disable_precomputed=False):
    """Load or compute the proposed method ranking."""
    scores = None

    if not disable_precomputed and network_name:
        try:
            precomputed = load_precomputed_rankings(network_name)
            if precomputed and proposed_method in precomputed and precomputed[proposed_method]:
                scores = precomputed[proposed_method]
                print(f"    ✓ Using precomputed ranking for {proposed_method}")
        except Exception as e:
            print(f"    ⚠ Failed to load precomputed ranking: {e}")

    if scores is None:
        print(f"    ⚠ Computing {proposed_method} scores on-the-fly")
        scores = get_node_scores(proposed_method, g)

    return rank_nodes_from_scores(scores)


def compute_beta(g, gamma, threshold_multiplier, use_gamma_scaled_threshold=False, cap_beta=True):
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    k_mean = float(np.mean(degrees))
    k2_mean = float(np.mean(degrees ** 2))
    denom = k2_mean - k_mean
    if denom <= 0:
        beta_th = 0.0
    else:
        beta_th = k_mean / denom
        if use_gamma_scaled_threshold:
            beta_th *= gamma
    beta = threshold_multiplier * beta_th
    if cap_beta:
        beta = min(beta, 1.0)
    return beta_th, beta, k_mean, k2_mean


# =========================================================
# 2. Discrete-time SIR evaluation
# =========================================================
def run_sir_once(g, seeds, beta, gamma, rng_seed=None, max_steps=1000):
    """
    Discrete-time synchronous network SIR.
    1. Infected nodes infect susceptible neighbors with probability beta.
    2. After infection attempts, infected nodes recover with probability gamma.
    3. Newly infected nodes become active in the next time step.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random

    infected = set(n for n in seeds if g.has_node(n))
    recovered = set()
    if not infected:
        return 0

    for _ in range(max_steps):
        if not infected:
            break
        new_infected = set()
        new_recovered = set()

        for u in list(infected):
            for v in g.neighbors(u):
                if v not in infected and v not in recovered:
                    if rng.random() < beta:
                        new_infected.add(v)
            if rng.random() < gamma:
                new_recovered.add(u)

        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)

    return len(infected) + len(recovered)


def make_seed_matrix(num_blocks, repeats_per_block, base_seed):
    return {(b, r): base_seed + b * 1000 + r for b in range(num_blocks) for r in range(repeats_per_block)}


def evaluate_seed_set_blocks(g, seeds, beta, gamma, seed_matrix, num_blocks, repeats_per_block):
    """Return mean final infection scale (%), 95% CI, and block-level means."""
    n = g.number_of_nodes()
    block_means = []

    for b in range(num_blocks):
        vals = []
        for r in range(repeats_per_block):
            impact = run_sir_once(g, seeds, beta, gamma, rng_seed=seed_matrix[(b, r)])
            vals.append(impact / n * 100.0)
        block_means.append(float(np.mean(vals)))

    block_means = np.asarray(block_means, dtype=float)
    mean_val = float(np.mean(block_means))
    if len(block_means) > 1:
        std_val = float(np.std(block_means, ddof=1))
        ci95 = float(t.ppf(0.975, df=len(block_means) - 1) * std_val / math.sqrt(len(block_means)))
    else:
        ci95 = 0.0
    return mean_val, ci95, block_means


# =========================================================
# 2.1 Paired statistics for matched-control comparisons
# =========================================================
def mean_ci95(values):
    """Return mean and two-sided 95% t-CI half width."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean_val = float(np.mean(values))
    if len(values) <= 1:
        return mean_val, 0.0
    sd = float(np.std(values, ddof=1))
    ci95 = float(
        t.ppf(0.975, df=len(values) - 1)
        * sd / math.sqrt(len(values))
    )
    return mean_val, ci95


def exact_signed_rank_pvalue_from_differences(d):
    """
    Exact two-sided signed-rank p-value by exhaustive sign enumeration.
    Used only when the number of non-zero paired differences is small.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 1.0

    ranks = rankdata(np.abs(d), method='average')
    w_plus_obs = float(np.sum(ranks[d > 0]))
    rank_sum = float(np.sum(ranks))
    center = rank_sum / 2.0
    obs_distance = abs(w_plus_obs - center)

    n = len(d)
    total = 1 << n
    extreme = 0

    for bits in range(total):
        w_plus = 0.0
        for i in range(n):
            if (bits >> i) & 1:
                w_plus += ranks[i]
        if abs(w_plus - center) >= obs_distance - 1e-12:
            extreme += 1

    return float(extreme / total)


def paired_wilcoxon_pvalue(x, y):
    """
    Two-sided paired Wilcoxon signed-rank test.

    Statistical unit here is one Monte Carlo block. For the default setting
    there are 50 paired block-level observations. If many zero differences
    reduce the effective sample size to <=20, exact sign enumeration is used;
    otherwise SciPy's normal approximation is used explicitly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    d_nonzero = d[d != 0]

    if len(d_nonzero) == 0:
        return 1.0

    if len(d_nonzero) <= 20:
        return exact_signed_rank_pvalue_from_differences(d_nonzero)

    return float(
        wilcoxon(
            d_nonzero,
            zero_method='wilcox',
            correction=False,
            alternative='two-sided',
            method='approx',
        ).pvalue
    )


def matched_pairs_rank_biserial(x, y):
    """
    Matched-pairs rank-biserial correlation:
        r_rb = (W+ - W-) / (W+ + W-)

    x = MSH block-level values
    y = control-strategy block-level values

    Positive r_rb favors MSH.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method='average')
    w_plus = float(np.sum(ranks[d > 0]))
    w_minus = float(np.sum(ranks[d < 0]))
    denom = w_plus + w_minus

    if denom == 0:
        return 0.0

    return float((w_plus - w_minus) / denom)


def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values; NaNs are preserved."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    ordered_p = p[order]
    m = len(order)

    adj = ordered_p * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    adjusted[order] = adj
    return adjusted


def apply_bh_by_control(df):
    """
    For each network and each control strategy separately, correct the
    paired-Wilcoxon p-values across the available seed-ratio settings.
    With complete data this is a family of 10 tests (p=1%,...,10%).
    """
    df = df.copy()
    df['P_BH'] = np.nan

    for control, sub in df.groupby('Control'):
        idx = sub.index.to_numpy()
        df.loc[idx, 'P_BH'] = bh_adjust(
            sub['Paired_Wilcoxon_P_raw'].to_numpy(dtype=float)
        )

    return df


# =========================================================
# 3. Degree-matched control generation
# =========================================================
def build_degree_quantile_bins(g, num_bins=10):
    """
    Assign nodes into degree-quantile bins.
    This avoids overly strict exact-degree matching while preserving the degree profile approximately.
    """
    nodes_sorted = sorted(g.nodes(), key=lambda n: (g.degree(n), safe_node_sort_key(n)))
    n = len(nodes_sorted)
    num_bins = max(1, min(num_bins, n))

    node_to_bin = {}
    bin_to_nodes = defaultdict(list)
    for idx, node in enumerate(nodes_sorted):
        b = min(num_bins - 1, int(idx * num_bins / n))
        node_to_bin[node] = b
        bin_to_nodes[b].append(node)

    return node_to_bin, bin_to_nodes


def sample_degree_matched_set(g, reference_seeds, node_to_bin, bin_to_nodes, rng):
    """Sample one unique seed set approximately matching the degree-bin profile of reference_seeds."""
    selected = []
    selected_set = set()
    all_nodes = list(g.nodes())

    # Shuffle reference nodes to reduce systematic collisions.
    refs = list(reference_seeds)
    rng.shuffle(refs)

    for ref in refs:
        b = node_to_bin.get(ref)
        candidates = []
        if b is not None:
            candidates = [n for n in bin_to_nodes[b] if n not in selected_set]

        if not candidates:
            candidates = [n for n in all_nodes if n not in selected_set]
        if not candidates:
            break

        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_set.add(chosen)

    # Defensive fill if something failed.
    while len(selected) < len(reference_seeds):
        candidates = [n for n in all_nodes if n not in selected_set]
        if not candidates:
            break
        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_set.add(chosen)

    return selected


def canonical_seed_tuple(seeds):
    return tuple(sorted(seeds, key=safe_node_sort_key))


def generate_degree_matched_candidates(g, reference_seeds, node_to_bin, bin_to_nodes,
                                       num_candidates, rng, max_attempt_factor=20):
    """Generate unique degree-matched candidate seed sets."""
    candidates = []
    seen = set()
    max_attempts = max(num_candidates * max_attempt_factor, num_candidates + 10)

    for _ in range(max_attempts):
        s = sample_degree_matched_set(g, reference_seeds, node_to_bin, bin_to_nodes, rng)
        if len(s) != len(reference_seeds):
            continue
        key = canonical_seed_tuple(s)
        if key not in seen:
            seen.add(key)
            candidates.append(s)
            if len(candidates) >= num_candidates:
                break

    if len(candidates) < num_candidates:
        print(f"      ⚠ Only generated {len(candidates)} unique controls, requested {num_candidates}")
    return candidates


def generate_degree_matched_candidates_incremental(g, reference_seeds, node_to_bin, bin_to_nodes,
                                                   num_new, rng, seen=None, max_attempt_factor=25):
    """
    Generate additional unique degree-matched candidate sets.

    This function is used only for DDMR caliper matching. It preserves an external
    `seen` set so that adaptive resampling does not repeatedly evaluate the same
    seed sets. DMR and DMD are still based on the initial candidate pool and are
    therefore unchanged by the adaptive DDMR search.
    """
    if seen is None:
        seen = set()

    candidates = []
    max_attempts = max(num_new * max_attempt_factor, num_new + 10)
    for _ in range(max_attempts):
        s = sample_degree_matched_set(g, reference_seeds, node_to_bin, bin_to_nodes, rng)
        if len(s) != len(reference_seeds):
            continue
        key = canonical_seed_tuple(s)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(s)
        if len(candidates) >= num_new:
            break
    return candidates, seen


def relative_ls_error(ls_control, ls_proposed, eps=1e-12):
    """Relative absolute L_s error used for DDMR caliper matching."""
    denom = max(abs(float(ls_proposed)), eps)
    return abs(float(ls_control) - float(ls_proposed)) / denom


def add_ls_error_to_info(info, ls_proposed):
    info = dict(info)
    info['Abs_Ls_Error'] = abs(float(info['Ls']) - float(ls_proposed))
    info['Rel_Ls_Error'] = relative_ls_error(info['Ls'], ls_proposed)
    return info


def select_ddmr_with_caliper(g, proposed_seeds, node_to_bin, bin_to_nodes,
                             initial_cand_info, ls_proposed, dist_calc, rng, args):
    """
    Degree-and-distance-matched random controls (DDMR).

    Matching requirement
    --------------------
        |L_s(C) - L_s(MSH)| / L_s(MSH) <= args.ddmr_caliper

    Final reviewer-oriented rule
    ----------------------------
    1. Start from the initial degree-matched candidate pool.
    2. If fewer than `ddmr_target_sets` valid matches are available,
       continue generating UNIQUE degree-matched candidates up to
       `ddmr_max_candidates`.
    3. If >= target valid matches are found, randomly sample exactly
       `ddmr_target_sets` valid controls.
    4. If only 1...(target-1) valid matches exist after the maximum
       search, evaluate ALL of those valid controls rather than deleting
       the whole setting.
    5. If zero valid matches exist, mark the setting as unmatched and
       omit only that DDMR point.
    6. No out-of-caliper fallback is ever used.

    This keeps every reported DDMR control strictly within the stated
    L_s tolerance while transparently reporting the actual number of
    valid and evaluated controls.
    """
    ddmr_pool = [
        add_ls_error_to_info(x, ls_proposed)
        for x in initial_cand_info
    ]

    seen = {
        canonical_seed_tuple(x["seeds"])
        for x in ddmr_pool
    }

    def valid_matches(pool):
        return [
            x for x in pool
            if x["Rel_Ls_Error"] <= args.ddmr_caliper
        ]

    matched = valid_matches(ddmr_pool)
    total_evaluated = len(ddmr_pool)

    target = int(args.ddmr_target_sets)

    while (
        len(matched) < target
        and total_evaluated < args.ddmr_max_candidates
    ):
        remaining = args.ddmr_max_candidates - total_evaluated
        batch_size = min(args.ddmr_batch_size, remaining)

        if batch_size <= 0:
            break

        new_candidates, seen = generate_degree_matched_candidates_incremental(
            g,
            proposed_seeds,
            node_to_bin,
            bin_to_nodes,
            num_new=batch_size,
            rng=rng,
            seen=seen,
        )

        if not new_candidates:
            break

        for cand in new_candidates:
            ls = dist_calc.average_distance(cand)
            ddmr_pool.append(
                add_ls_error_to_info(
                    {"seeds": cand, "Ls": ls},
                    ls_proposed,
                )
            )

        total_evaluated = len(ddmr_pool)
        matched = valid_matches(ddmr_pool)

    # Final selection rule.
    if len(matched) >= target:
        selected = rng.sample(matched, target)
        status = "matched_target_reached"
    elif len(matched) > 0:
        selected = list(matched)
        status = "matched_partial_use_all_valid"
    else:
        selected = []
        status = "unmatched_no_valid_control"

    best_rel_error = min(
        (x["Rel_Ls_Error"] for x in ddmr_pool),
        default=np.nan,
    )

    selected_rel_errors = [
        x["Rel_Ls_Error"]
        for x in selected
    ]

    selected_mean_rel_error = (
        float(np.mean(selected_rel_errors))
        if selected_rel_errors else np.nan
    )

    selected_max_rel_error = (
        float(np.max(selected_rel_errors))
        if selected_rel_errors else np.nan
    )

    acceptance_rate = (
        len(matched) / total_evaluated * 100.0
        if total_evaluated > 0 else np.nan
    )

    meta = {
        "DDMR_Caliper(%)": args.ddmr_caliper * 100.0,
        "DDMR_Target_Sets": target,
        "DDMR_Status": status,
        "DDMR_Total_Candidates_Evaluated": total_evaluated,
        "DDMR_Candidates_Within_Caliper": len(matched),
        "DDMR_Acceptance_Rate(%)": acceptance_rate,
        "DDMR_Selected_Count": len(selected),
        "DDMR_Best_Relative_Ls_Error(%)":
            best_rel_error * 100.0
            if np.isfinite(best_rel_error) else np.nan,
        "DDMR_Selected_Mean_Relative_Ls_Error(%)":
            selected_mean_rel_error * 100.0
            if np.isfinite(selected_mean_rel_error) else np.nan,
        "DDMR_Selected_Max_Relative_Ls_Error(%)":
            selected_max_rel_error * 100.0
            if np.isfinite(selected_max_rel_error) else np.nan,
    }

    return selected, meta


# =========================================================
# 4. Spatial dispersion L_s
# =========================================================
class DistanceCalculator:
    """Average shortest-path distance among selected seeds with caching and pair sampling."""
    def __init__(self, g, max_exact_pairs=6000, sample_pairs=6000, seed=42):
        self.g = g
        self.max_exact_pairs = max_exact_pairs
        self.sample_pairs = sample_pairs
        self.rng = random.Random(seed)
        self.spl_cache = {}

        if nx.is_connected(g):
            try:
                self.disconnected_distance = nx.diameter(g) + 1
            except Exception:
                self.disconnected_distance = g.number_of_nodes()
        else:
            try:
                largest_cc = max(nx.connected_components(g), key=len)
                self.disconnected_distance = nx.diameter(g.subgraph(largest_cc)) + 1
            except Exception:
                self.disconnected_distance = g.number_of_nodes()

    def dist(self, u, v):
        if u == v:
            return 0
        if u not in self.spl_cache:
            self.spl_cache[u] = nx.single_source_shortest_path_length(self.g, u)
        return self.spl_cache[u].get(v, self.disconnected_distance)

    def average_distance(self, seeds):
        seeds = [s for s in seeds if self.g.has_node(s)]
        k = len(seeds)
        if k <= 1:
            return 0.0

        total_pairs = k * (k - 1) // 2
        if total_pairs <= self.max_exact_pairs:
            pairs_iter = combinations(seeds, 2)
            total = 0.0
            count = 0
            for u, v in pairs_iter:
                total += self.dist(u, v)
                count += 1
            return total / count if count else 0.0

        # Approximate by sampled pairs for large seed sets.
        count = min(self.sample_pairs, total_pairs)
        total = 0.0
        sampled = set()
        attempts = 0
        max_attempts = count * 20
        while len(sampled) < count and attempts < max_attempts:
            i = self.rng.randrange(k)
            j = self.rng.randrange(k)
            if i == j:
                attempts += 1
                continue
            if i > j:
                i, j = j, i
            if (i, j) in sampled:
                attempts += 1
                continue
            sampled.add((i, j))
            total += self.dist(seeds[i], seeds[j])
            attempts += 1

        if not sampled:
            return 0.0
        return total / len(sampled)


# =========================================================
# 5. Network-level experiment
# =========================================================
def process_one_network(net, args, output_dir):
    print(f"\n[Network] {net}")
    print("-" * 60)

    g = download_and_load_graph(net, verbose=False)
    if g is None or g.number_of_nodes() == 0:
        print(f"  [Skip] {net}: empty or failed to load")
        return None

    n = g.number_of_nodes()
    m = g.number_of_edges()
    print(f"  Nodes: {n}, Edges: {m}")

    proposed_ranked = get_proposed_ranking(
        g,
        net,
        proposed_method=args.proposed_method,
        disable_precomputed=args.disable_precomputed,
    )
    proposed_ranked = [
        v for v in proposed_ranked
        if g.has_node(v)
    ]

    beta_th, beta, k_mean, k2_mean = compute_beta(
        g,
        gamma=args.gamma,
        threshold_multiplier=args.threshold_multiplier,
        use_gamma_scaled_threshold=args.use_gamma_scaled_threshold,
        cap_beta=not args.no_cap_beta,
    )

    print(
        f"  SIR: gamma={args.gamma:.3f}, "
        f"beta_th={beta_th:.6f}, beta={beta:.6f}, "
        f"lambda={args.threshold_multiplier:.2f}"
    )

    node_to_bin, bin_to_nodes = build_degree_quantile_bins(
        g,
        num_bins=args.degree_bins
    )

    dist_calc = DistanceCalculator(
        g,
        max_exact_pairs=args.max_exact_pairs,
        sample_pairs=args.sample_pairs,
        seed=args.master_seed + stable_int_hash(net, 100000),
    )

    records = []

    for ratio_pct in tqdm(
        range(1, 11),
        desc=f"  Ratios for {net}"
    ):
        ratio = ratio_pct / 100.0
        k = max(2, int(math.ceil(n * ratio)))
        k = min(k, n)

        proposed_seeds = proposed_ranked[:k]
        ls_proposed = dist_calc.average_distance(
            proposed_seeds
        )

        # -------------------------------------------------
        # Initial degree-matched candidate pool
        # -------------------------------------------------
        control_rng = random.Random(
            args.master_seed
            + stable_int_hash(
                f'{net}_{ratio_pct}',
                1000000
            )
        )

        candidates = generate_degree_matched_candidates(
            g,
            proposed_seeds,
            node_to_bin,
            bin_to_nodes,
            num_candidates=args.control_trials,
            rng=control_rng,
        )

        if not candidates:
            print(
                f"    [Skip] {net} p={ratio_pct}%: "
                f"no degree-matched candidates"
            )
            continue

        cand_info = []

        for cand in candidates:
            cand_info.append({
                'seeds': cand,
                'Ls': dist_calc.average_distance(cand),
            })

        initial_candidate_count = len(cand_info)

        # -------------------------------------------------
        # DMR: 10 randomly selected unique degree-matched sets
        # -------------------------------------------------
        dmr_pool = list(cand_info)
        control_rng.shuffle(dmr_pool)
        dmr_infos = dmr_pool[
            :min(
                args.dmr_eval_sets,
                len(dmr_pool)
            )
        ]

        # -------------------------------------------------
        # DMD: single maximum-dispersion set from same pool
        # -------------------------------------------------
        dmd_info = max(
            cand_info,
            key=lambda x: x['Ls']
        )

        # -------------------------------------------------
        # DDMR: target 10 random valid controls within 5%
        # -------------------------------------------------
        ddmr_rng = random.Random(
            args.master_seed
            + 777777
            + ratio_pct * 100000
            + stable_int_hash(net, 10000)
        )

        ddmr_infos, ddmr_meta = select_ddmr_with_caliper(
            g,
            proposed_seeds,
            node_to_bin,
            bin_to_nodes,
            initial_cand_info=cand_info,
            ls_proposed=ls_proposed,
            dist_calc=dist_calc,
            rng=ddmr_rng,
            args=args,
        )

        # -------------------------------------------------
        # Common random-number schedule
        # -------------------------------------------------
        seed_matrix = make_seed_matrix(
            args.blocks,
            args.repeats,
            base_seed=(
                args.master_seed
                + ratio_pct * 100000
                + stable_int_hash(net, 10000)
            ),
        )

        f_proposed, ci_proposed, blocks_proposed = (
            evaluate_seed_set_blocks(
                g,
                proposed_seeds,
                beta,
                args.gamma,
                seed_matrix,
                args.blocks,
                args.repeats,
            )
        )

        # -------------------------------------------------
        # Strategy-level evaluation
        #
        # For DMR/DDMR, evaluate each control set using the
        # same block/repetition RNG schedule, then average the
        # control sets WITHIN each block. CI and paired tests are
        # based on the resulting 50 strategy-level block means.
        # -------------------------------------------------
        def eval_control_group(infos):
            if not infos:
                return None

            ls_vals = []
            block_matrix = []

            for info in infos:
                _, _, block_means = evaluate_seed_set_blocks(
                    g,
                    info['seeds'],
                    beta,
                    args.gamma,
                    seed_matrix,
                    args.blocks,
                    args.repeats,
                )

                ls_vals.append(info['Ls'])
                block_matrix.append(
                    np.asarray(block_means, dtype=float)
                )

            block_matrix = np.vstack(block_matrix)

            # One strategy-level value per Monte Carlo block.
            strategy_blocks = np.mean(
                block_matrix,
                axis=0
            )

            f_mean, f_ci = mean_ci95(
                strategy_blocks
            )

            return {
                'F': f_mean,
                'CI': f_ci,
                'Ls': float(np.mean(ls_vals)),
                'BlockMeans': strategy_blocks,
                'Control_Set_Count': len(infos),
            }

        control_results = {
            'DMR': eval_control_group(dmr_infos),
            'DMD': eval_control_group([dmd_info]),
        }

        if ddmr_infos:
            control_results['DDMR'] = eval_control_group(
                ddmr_infos
            )

        # -------------------------------------------------
        # Matching metadata: repeated in row-level figure data;
        # later deduplicated into Supplementary Table Sx.
        # -------------------------------------------------
        base_meta = {
            'Network': net,
            'N': n,
            'E': m,
            'p(%)': ratio_pct,
            'k': k,
            'gamma': args.gamma,
            'lambda': args.threshold_multiplier,
            'beta_th': beta_th,
            'beta': beta,
            f'{args.proposed_label}_F(%)': f_proposed,
            f'{args.proposed_label}_95CI': ci_proposed,
            f'{args.proposed_label}_Ls': ls_proposed,

            'Initial_DegreeMatched_Candidates':
                initial_candidate_count,
            'DMR_Sets_Evaluated':
                len(dmr_infos),
            'DMD_Candidate_Pool_Size':
                initial_candidate_count,
            'DMD_Sets_Evaluated':
                1,
        }

        base_meta.update(ddmr_meta)

        # -------------------------------------------------
        # Quantitative paired comparisons
        # -------------------------------------------------
        for control_name, cres in control_results.items():
            if cres is None:
                continue

            control_blocks = np.asarray(
                cres['BlockMeans'],
                dtype=float
            )

            delta_blocks = (
                np.asarray(blocks_proposed, dtype=float)
                - control_blocks
            )

            delta_mean, delta_ci = mean_ci95(
                delta_blocks
            )

            p_raw = paired_wilcoxon_pvalue(
                blocks_proposed,
                control_blocks,
            )

            r_rb = matched_pairs_rank_biserial(
                blocks_proposed,
                control_blocks,
            )

            rec = dict(base_meta)

            rec.update({
                'Control': control_name,
                'Control_Sets_Evaluated':
                    cres['Control_Set_Count'],

                'Control_F(%)':
                    cres['F'],
                'Control_95CI':
                    cres['CI'],
                'Control_Ls':
                    cres['Ls'],

                'Delta_Ls_Control_minus_Proposed':
                    cres['Ls'] - ls_proposed,

                'Delta_F_Proposed_minus_Control(%)':
                    delta_mean,
                'Delta_F_95CI':
                    delta_ci,

                'Paired_Wilcoxon_P_raw':
                    p_raw,
                'Rank_Biserial_r_rb':
                    r_rb,
            })

            records.append(rec)

        # -------------------------------------------------
        # Concise console diagnostics
        # -------------------------------------------------
        ddmr_msg = (
            f"DDMR valid={ddmr_meta['DDMR_Candidates_Within_Caliper']}, "
            f"evaluated={ddmr_meta['DDMR_Selected_Count']}, "
            f"searched={ddmr_meta['DDMR_Total_Candidates_Evaluated']} "
            f"[{ddmr_meta['DDMR_Status']}]"
            if ddmr_infos
            else
            f"DDMR unmatched: "
            f"{ddmr_meta['DDMR_Candidates_Within_Caliper']} valid "
            f"after {ddmr_meta['DDMR_Total_Candidates_Evaluated']} searched"
        )

        print(
            f"    p={ratio_pct:2d}% k={k:4d} | "
            f"{args.proposed_label} F={f_proposed:6.2f}, "
            f"Ls={ls_proposed:5.2f} | "
            f"DMR={len(dmr_infos)} sets | "
            f"DMD=1/{initial_candidate_count} | "
            f"{ddmr_msg}"
        )

    if not records:
        return None

    df_net = pd.DataFrame(records)

    # BH correction is performed separately for DMR, DMD, and DDMR
    # across the seed-ratio comparisons within this network.
    df_net = apply_bh_by_control(df_net)

    plot_single_network_quadrant(
        df_net,
        net,
        args.proposed_label,
        output_dir,
        annotate=args.annotate_ratios,
    )

    return df_net


# =========================================================
# 6. Summary and plotting
# =========================================================
def summarize_network(df, proposed_label='MSH'):
    rows = []
    for control, sub in df.groupby('Control'):
        rows.append({
            'Control': control,
            'Mean_Delta_Ls': sub['Delta_Ls_Control_minus_Proposed'].mean(),
            'Mean_Delta_F(%)': sub['Delta_F_Proposed_minus_Control(%)'].mean(),
            'Mean_Imp_vs_Control(%)': sub['Imp_vs_Control(%)'].mean(),
            'Cases_Control_More_Dispersed(%)': (sub['Delta_Ls_Control_minus_Proposed'] > 0).mean() * 100.0,
            f'Cases_{proposed_label}_Higher_F(%)': (sub['Delta_F_Proposed_minus_Control(%)'] > 0).mean() * 100.0,
            'Upper_Right_Cases(%)': ((sub['Delta_Ls_Control_minus_Proposed'] > 0) & (sub['Delta_F_Proposed_minus_Control(%)'] > 0)).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def add_zero_axes(ax):
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.75, zorder=1)
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.75, zorder=1)


def finish_axes_style(ax):
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)


def plot_single_network_quadrant(df_net, net, proposed_label, output_dir, annotate=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    add_zero_axes(ax)

    for control_name, style in CONTROL_STYLES.items():
        sub = (
            df_net[df_net['Control'] == control_name]
            .sort_values('p(%)')
        )

        if sub.empty:
            continue

        x = sub[
            'Delta_Ls_Control_minus_Proposed'
        ].to_numpy(dtype=float)

        y = sub[
            'Delta_F_Proposed_minus_Control(%)'
        ].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=34,
            marker=style['marker'],
            facecolor=style['color'],
            edgecolor='black',
            linewidth=0.45,
            alpha=0.88,
            label=style['label'],
            zorder=5,
        )

        if annotate:
            for _, row in sub.iterrows():
                ax.text(
                    row[
                        'Delta_Ls_Control_minus_Proposed'
                    ],
                    row[
                        'Delta_F_Proposed_minus_Control(%)'
                    ],
                    str(int(row['p(%)'])),
                    fontsize=6.5,
                    ha='center',
                    va='bottom',
                )

    ax.set_title(
        format_network_name(net),
        fontsize=12,
        fontweight='bold',
        pad=8,
    )

    ax.set_xlabel(r'$\Delta L_s$')
    ax.set_ylabel(r'$\Delta F(t_c)$ (pp)')
    finish_axes_style(ax)

    plt.tight_layout(pad=0.25)

    pdf_path = os.path.join(
        output_dir,
        f"Opinion10_DDMR_Quadrant_{net}.pdf"
    )
    png_path = os.path.join(
        output_dir,
        f"Opinion10_DDMR_Quadrant_{net}.png"
    )

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
    )

    plt.close()

    print(
        f"  [Output] Network plot saved: "
        f"{pdf_path}"
    )


def export_standalone_legend(output_dir):
    fig, ax = plt.subplots(figsize=(7.2, 0.85))
    ax.axis('off')
    handles = []
    for control_name, style in CONTROL_STYLES.items():
        handles.append(Line2D(
            [0], [0],
            marker=style['marker'],
            color='none',
            markerfacecolor=style['color'],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=7.0,
            label=style['label'],
        ))
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=3,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='black',
        framealpha=1.0,
        columnspacing=1.2,
        handletextpad=0.45,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.8)
    pdf_path = os.path.join(output_dir, "Opinion10_DDMR_Quadrant_Legend_Standalone.pdf")
    png_path = os.path.join(output_dir, "Opinion10_DDMR_Quadrant_Legend_Standalone.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_9network_panel(df_all, proposed_label, output_dir, network_order=None, annotate=False):
    if network_order is None:
        network_order = list(
            df_all['Network'].drop_duplicates()
        )

    network_order = network_order[:9]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.5, 8.4),
        sharex=False,
        sharey=False,
    )

    axes = axes.flatten()

    for ax, net in zip(
        axes,
        network_order
    ):
        sub_net = df_all[
            df_all['Network'] == net
        ]

        add_zero_axes(ax)

        for control_name, style in CONTROL_STYLES.items():
            sub = (
                sub_net[
                    sub_net['Control'] == control_name
                ]
                .sort_values('p(%)')
            )

            if sub.empty:
                continue

            x = sub[
                'Delta_Ls_Control_minus_Proposed'
            ].to_numpy(dtype=float)

            y = sub[
                'Delta_F_Proposed_minus_Control(%)'
            ].to_numpy(dtype=float)

            ax.scatter(
                x,
                y,
                s=24,
                marker=style['marker'],
                facecolor=style['color'],
                edgecolor='black',
                linewidth=0.35,
                alpha=0.86,
                zorder=5,
            )

            if annotate:
                for _, row in sub.iterrows():
                    ax.text(
                        row[
                            'Delta_Ls_Control_minus_Proposed'
                        ],
                        row[
                            'Delta_F_Proposed_minus_Control(%)'
                        ],
                        str(int(row['p(%)'])),
                        fontsize=5.8,
                        ha='center',
                        va='bottom',
                    )

        ax.set_title(
            format_network_name(net),
            fontsize=12,
            fontweight='bold',
            pad=8,
        )

        finish_axes_style(ax)

    for ax in axes[len(network_order):]:
        ax.axis('off')

    fig.supxlabel(
        r'$\Delta L_s$',
        y=0.045,
        fontsize=11,
    )

    fig.supylabel(
        r'$\Delta F(t_c)$ (pp)',
        x=0.045,
        fontsize=11,
    )

    handles = []

    for control_name, style in CONTROL_STYLES.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style['marker'],
                color='none',
                markerfacecolor=style['color'],
                markeredgecolor='black',
                markeredgewidth=0.6,
                markersize=6.5,
                label=style['label'],
            )
        )

    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.005),
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=1.0,
        fontsize=9,
    )

    plt.tight_layout(
        rect=[0.06, 0.06, 1.0, 0.955]
    )

    pdf_path = os.path.join(
        output_dir,
        "Opinion10_DDMR_Quadrant_9Networks.pdf"
    )

    png_path = os.path.join(
        output_dir,
        "Opinion10_DDMR_Quadrant_9Networks.png"
    )

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
    )

    plt.close()

    print(
        f"[Output] 9-network panel saved: "
        f"{pdf_path}"
    )


def save_all_results(df_all, output_dir, proposed_label):
    """
    Save only reviewer-relevant outputs:

    Figure_Data
        One row = one seed ratio × one control strategy.
        Contains the coordinates used in Fig. 9; paired uncertainty is retained for audit and Table Sy.

    Table_Sx_Matching
        Candidate-generation and matching diagnostics.

    Table_Sy_Statistics
        Mean performance, paired ΔF with 95% CI, BH-adjusted p,
        and matched-pairs rank-biserial effect size.

    Presentation choice
    -------------------
    Fig. 9 is intentionally kept as a clean quadrant scatter plot.
    The paired 95% CIs are not drawn on the scatter; they remain fully
    reported in Table_Sy_Statistics.
    """
    data_path = os.path.join(
        output_dir,
        "Opinion10_DDMR_Quadrant_Data.xlsx"
    )

    # -----------------------------------------------------
    # Supplementary Table Sx: matching construction
    # one row per network × seed ratio
    # -----------------------------------------------------
    matching_cols = [
        'Network',
        'p(%)',
        'k',
        'Initial_DegreeMatched_Candidates',
        'DMR_Sets_Evaluated',
        'DMD_Candidate_Pool_Size',
        'DMD_Sets_Evaluated',
        'DDMR_Caliper(%)',
        'DDMR_Target_Sets',
        'DDMR_Total_Candidates_Evaluated',
        'DDMR_Candidates_Within_Caliper',
        'DDMR_Acceptance_Rate(%)',
        'DDMR_Selected_Count',
        'DDMR_Selected_Mean_Relative_Ls_Error(%)',
        'DDMR_Selected_Max_Relative_Ls_Error(%)',
        'DDMR_Status',
    ]

    table_sx = (
        df_all[
            [c for c in matching_cols if c in df_all.columns]
        ]
        .drop_duplicates(
            subset=['Network', 'p(%)']
        )
        .sort_values(
            ['Network', 'p(%)']
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Supplementary Table Sy: quantitative comparisons
    # -----------------------------------------------------
    table_sy = df_all[
        [
            'Network',
            'p(%)',
            'Control',
            'Control_Sets_Evaluated',
            f'{proposed_label}_F(%)',
            f'{proposed_label}_95CI',
            'Control_F(%)',
            'Control_95CI',
            'Delta_F_Proposed_minus_Control(%)',
            'Delta_F_95CI',
            'P_BH',
            'Rank_Biserial_r_rb',
        ]
    ].copy()

    table_sy = table_sy.rename(
        columns={
            f'{proposed_label}_F(%)':
                f'{proposed_label}_F_mean(%)',
            f'{proposed_label}_95CI':
                f'{proposed_label}_95CI_half',
            'Control_95CI':
                'Control_95CI_half',
            'Delta_F_Proposed_minus_Control(%)':
                'Delta_F(pp)',
            'Delta_F_95CI':
                'Delta_F_95CI_half(pp)',
            'P_BH':
                'p',
            'Rank_Biserial_r_rb':
                'r_rb',
        }
    )

    table_sy = table_sy.sort_values(
        ['Network', 'p(%)', 'Control']
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Figure data: transparent audit of every plotted point
    # -----------------------------------------------------
    figure_cols = [
        'Network',
        'p(%)',
        'Control',
        'Control_Sets_Evaluated',
        f'{proposed_label}_Ls',
        'Control_Ls',
        'Delta_Ls_Control_minus_Proposed',
        f'{proposed_label}_F(%)',
        'Control_F(%)',
        'Delta_F_Proposed_minus_Control(%)',
        'Delta_F_95CI',
        'P_BH',
        'Rank_Biserial_r_rb',
    ]

    figure_data = df_all[
        [c for c in figure_cols if c in df_all.columns]
    ].copy()

    with pd.ExcelWriter(
        data_path,
        engine='openpyxl'
    ) as writer:
        figure_data.to_excel(
            writer,
            sheet_name='Figure_Data',
            index=False,
        )

        table_sx.to_excel(
            writer,
            sheet_name='Table_Sx_Matching',
            index=False,
        )

        table_sy.to_excel(
            writer,
            sheet_name='Table_Sy_Statistics',
            index=False,
        )

    print(
        f"[Output] Combined reviewer tables saved: "
        f"{data_path}"
    )


# =========================================================
# 7. CLI and main workflow
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Opinion 10 DDMR quadrant scatter experiment on nine networks')
    parser.add_argument('--networks', type=str, default='',
                        help='Comma-separated network names. Default: get_network_list().')
    parser.add_argument('--output-dir', type=str,
                        default='results/exp_opinion10_ddmr_quadrant_9networks_caliper5')

    parser.add_argument('--proposed-method', type=str, default='HOSH',
                        help='Implementation method name, usually HOSH.')
    parser.add_argument('--proposed-label', type=str, default='MSH',
                        help='Display label in figures/tables, e.g., MSH.')
    parser.add_argument('--disable-precomputed', action='store_true',
                        help='Compute proposed scores directly instead of loading precomputed rankings.')

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--threshold-multiplier', type=float, default=2.5)
    parser.add_argument('--use-gamma-scaled-threshold', action='store_true',
                        help='Use beta_th = gamma * <k>/(<k^2>-<k>). Default is unscaled threshold.')
    parser.add_argument('--no-cap-beta', action='store_true',
                        help='Do not cap beta at 1.0.')

    parser.add_argument('--blocks', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--control-trials', type=int, default=1000,
                        help='Number of degree-matched random candidate seed sets.')
    parser.add_argument('--dmr-eval-sets', type=int, default=10,
                        help='Number of DMR seed sets to evaluate and average.')
    parser.add_argument('--ddmr-target-sets', type=int, default=10,
                        help='Target number of valid DDMR control sets per network-ratio setting. '
                             'If only 1-9 valid controls exist after the maximum search, all are used.')
    parser.add_argument('--ddmr-caliper', type=float, default=0.05,
                        help='Relative L_s matching caliper for DDMR. Default: 0.05 means <= 5%% error.')
    parser.add_argument('--ddmr-max-candidates', type=int, default=10000,
                        help='Maximum unique degree-matched candidates evaluated for DDMR caliper matching.')
    parser.add_argument('--ddmr-batch-size', type=int, default=1000,
                        help='Batch size for adaptive DDMR candidate generation.')
    parser.add_argument('--degree-bins', type=int, default=10)

    parser.add_argument('--max-exact-pairs', type=int, default=6000,
                        help='Exact pair-distance computation threshold.')
    parser.add_argument('--sample-pairs', type=int, default=6000,
                        help='Sampled pairs for L_s when the seed set is large.')
    parser.add_argument('--master-seed', type=int, default=42)
    parser.add_argument('--annotate-ratios', action='store_true',
                        help='Annotate scatter points with seed ratio labels 1-10.')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.master_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Opinion 10: DDMR Quadrant Scatter Experiment on Nine Networks")
    print("=" * 72)
    print(f"Output directory: {args.output_dir}")
    print(f"Proposed method: implementation={args.proposed_method}, label={args.proposed_label}")
    print(f"SIR: gamma={args.gamma}, lambda={args.threshold_multiplier}, blocks={args.blocks}, repeats={args.repeats}")
    print(
        f"Controls: trials={args.control_trials}, DMR sets={args.dmr_eval_sets}, "
        f"DDMR target={args.ddmr_target_sets}, DDMR caliper={args.ddmr_caliper*100:.1f}%, "
        f"DDMR max candidates={args.ddmr_max_candidates}"
    )

    export_standalone_legend(args.output_dir)

    if args.networks.strip():
        networks = [x.strip() for x in args.networks.split(',') if x.strip()]
    else:
        networks = get_network_list()

    all_dfs = []
    processed_order = []

    for idx, net in enumerate(networks, 1):
        print(f"\n[{idx}/{len(networks)}]")
        try:
            df_net = process_one_network(net, args, args.output_dir)
            if df_net is not None and not df_net.empty:
                all_dfs.append(df_net)
                processed_order.append(net)

                # Combined checkpoint after each finished network.
                df_ckpt = pd.concat(all_dfs, ignore_index=True)
                ckpt_path = os.path.join(args.output_dir, "Opinion10_DDMR_Quadrant_Checkpoint.xlsx")
                df_ckpt.to_excel(ckpt_path, index=False)
                print(f"[Checkpoint] Combined checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"[Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        save_all_results(df_all, args.output_dir, args.proposed_label)
        plot_9network_panel(df_all, args.proposed_label, args.output_dir,
                            network_order=processed_order, annotate=args.annotate_ratios)
    else:
        warnings.warn("No network produced valid results.")

    print("\n" + "=" * 72)
    print("Completed.")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 72)


if __name__ == '__main__':
    main()
