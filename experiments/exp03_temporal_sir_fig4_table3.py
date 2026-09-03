"""
Experiment 6: Temporal SIR propagation experiment (review-response revision)

Purpose
-------
- Select Top-K ranked nodes as the initial infected seed set.
- Track the temporal evolution of cumulative infection scale F(t).
- Keep the temporal figure visually clean by plotting mean trajectories only.
- Perform formal statistical comparison only at the final time point.

Statistical protocol
--------------------
1) 50 independent Monte Carlo blocks;
2) 20 stochastic SIR realizations per block;
3) the ranked seed set is fixed for each method/network setting;
4) corresponding methods use the same method-independent pseudo-random seed schedule;
5) the 20 realizations within each block are averaged first;
6) 95% confidence intervals are computed from the 50 block means using Student's t distribution;
7) MSH (internal method key: HOSH) is compared with the strongest baseline at the final time
   using a paired Wilcoxon signed-rank test on the 50 paired block means;
8) matched-pairs rank-biserial correlation is reported as the standardized effect size;
9) final-time p-values are Benjamini-Hochberg adjusted across the nine network-level comparisons.

Output policy
-------------
- Main temporal figures: mean trajectories only (no CI bands, no p-value time series).
- Per-network Excel file:
    * Temporal_Mean_CI: only mean and 95% CI half-width for each method;
    * Final_Block_Pairs: the 50 paired final-time block observations used in the Wilcoxon test.
- Global summary Excel file:
    * only the quantities required for the manuscript/reviewer response:
      network, strongest baseline, MSH/baseline final means and 95% CIs,
      absolute difference, raw p-value, BH-adjusted p-value, rank-biserial effect size.

Notes
-----
- The codebase still uses the internal method name 'HOSH'; all exported labels use 'MSH'.
- No time-point-by-time-point significance testing is performed, avoiding unnecessary multiple testing.
"""

import os
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import t, wilcoxon, rankdata

from hosh_methods import get_node_scores, get_network_partition
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes


# ==========================================
# 0. Constants and plot configuration
# ==========================================
TARGET_METHOD = 'HOSH'      # internal code key
TARGET_DISPLAY = 'MSH'      # manuscript/output label
NUM_BLOCKS = 50
REPEATS_PER_BLOCK = 20
COMMUNITY_SEED = 42

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4.5,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Init] Global seed set to: {seed}")


def stable_int_hash(*items, modulo=2**32 - 1):
    """Stable integer hash for deterministic seed derivation across Python processes."""
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def display_method_name(method_name):
    """Use MSH in figures/tables while retaining HOSH as the internal code key."""
    return TARGET_DISPLAY if method_name == TARGET_METHOD else method_name


def format_network_name(name):
    s = str(name)
    return s[:1].upper() + s[1:].lower() if s else s


# ==========================================
# 1. Statistical utilities
# ==========================================
def make_seed_matrix(num_blocks=NUM_BLOCKS, repeats_per_block=REPEATS_PER_BLOCK, base_seed=2026):
    """
    Generate a method-independent pseudo-random seed schedule.

    For a fixed network/seed-budget setting, every method uses the same seed for the
    same (block, repeat) index. This provides a paired block design without claiming
    identical edge-level transmission realizations.
    """
    return {
        (b, r): base_seed + b * 1000 + r
        for b in range(num_blocks)
        for r in range(repeats_per_block)
    }


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR adjustment; returns adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    valid = ~np.isnan(p_values)
    if not np.any(valid):
        return q_values

    p = p_values[valid]
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)

    q_valid = np.empty_like(p)
    q_valid[order] = ranked_q
    q_values[valid] = q_valid
    return q_values


def rank_biserial_paired(x, y):
    """
    Matched-pairs rank-biserial correlation aligned with the paired Wilcoxon test.

    r_rb = (W+ - W-) / (W+ + W-)

    Positive values favor MSH (x), negative values favor the baseline (y).
    Zero paired differences are excluded, consistent with Wilcoxon's default zero method.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    if diff.size == 0:
        return 0.0

    ranks = rankdata(np.abs(diff), method='average')
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    denom = w_plus + w_minus
    return (w_plus - w_minus) / denom if denom > 0 else 0.0


def format_p_value(p):
    if pd.isna(p):
        return ''
    return f'{p:.2e}' if p < 0.001 else f'{p:.3f}'


# ==========================================
# 2. Plot utilities
# ==========================================
def export_standalone_legend(methods, colors, markers, output_dir):
    """Export a standalone legend to avoid overcrowding the temporal panels."""
    fig, ax = plt.subplots(figsize=(8.2, 1.2))
    ax.axis('off')

    handles = []
    for m in methods:
        line, = ax.plot(
            [], [], label=display_method_name(m),
            color=colors.get(m, '#000000'),
            marker=markers.get(m, 'o'),
            linestyle='--', linewidth=1.5, markersize=5,
            markerfacecolor=colors.get(m, '#000000'),
            markeredgecolor='black', markeredgewidth=0.5,
        )
        handles.append(line)

    ax.legend(
        handles=handles, loc='center', ncol=6,
        frameon=True, fancybox=False, shadow=False,
        edgecolor='black', fontsize=10,
    )

    pdf_path = os.path.join(output_dir, 'Temporal_Legend_Standalone.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close()


# ==========================================
# 3. Temporal SIR model
# ==========================================
def run_sir_temporal(graph, seeds, beta, gamma, max_steps=100, rng_seed=None):
    """
    Run one discrete-time synchronous SIR simulation and record cumulative infection size.

    Update convention:
    1. Each infected node attempts to infect each susceptible neighbor with probability beta.
    2. After infection attempts, each infected node recovers with probability gamma.
    3. Newly infected nodes become active in the next time step.
    4. F(t) records infected + recovered nodes at the beginning of each time step.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    infected_nodes = set(n for n in seeds if graph.has_node(n))
    recovered_nodes = set()

    if not infected_nodes:
        return [0] * max_steps

    temporal_infection = []

    for step in range(max_steps):
        total_infected = len(infected_nodes) + len(recovered_nodes)
        temporal_infection.append(total_infected)

        if not infected_nodes:
            temporal_infection.extend([total_infected] * (max_steps - step - 1))
            break

        new_infected = set()
        new_recovered = set()

        for u in list(infected_nodes):
            for v in graph.neighbors(u):
                if v not in infected_nodes and v not in recovered_nodes:
                    if rng.random() < beta:
                        new_infected.add(v)

            if rng.random() < gamma:
                new_recovered.add(u)

        infected_nodes.update(new_infected)
        infected_nodes.difference_update(new_recovered)
        recovered_nodes.update(new_recovered)

    if len(temporal_infection) < max_steps:
        last_val = temporal_infection[-1] if temporal_infection else 0
        temporal_infection.extend([last_val] * (max_steps - len(temporal_infection)))

    return temporal_infection[:max_steps]


def exp_temporal_sir(methods, g, top_k=10, network_name=None,
                     num_blocks=NUM_BLOCKS, repeats_per_block=REPEATS_PER_BLOCK,
                     base_seed=2026):
    """Temporal SIR propagation experiment with 50 block-level observations."""
    print('  [Exp: Temporal SIR] Running propagation analysis...')

    n_nodes = g.number_of_nodes()
    degrees = [d for _, d in g.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d ** 2 for d in degrees])

    gamma = 1.0
    threshold_multiplier = 2.5
    beta_th = gamma * (k_mean / (k2_mean - k_mean)) if (k2_mean - k_mean) > 0 else 0.0
    beta = threshold_multiplier * beta_th

    top_k = min(top_k, n_nodes)
    max_steps = 100 if n_nodes > 3000 else 50

    print(f'    Graph Properties: <k>={k_mean:.2f}, <k^2>={k2_mean:.2f}')
    print(f'    Epidemic Threshold: beta_th={beta_th:.6f}')
    print(f'    SIR Parameters: beta={beta:.6f} ({threshold_multiplier:g}×beta_th), gamma={gamma:.2f}')
    print(f'    Seed Count: {top_k} nodes ({top_k / n_nodes * 100:.2f}% of network)')
    print(f'    Monte Carlo design: {num_blocks} blocks × {repeats_per_block} realizations/block')

    seed_matrix = make_seed_matrix(num_blocks, repeats_per_block, base_seed)
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    # Keep CHBC reproducible if a precomputed ranking is unavailable.
    global_partition, global_comm_sizes = None, None
    if 'CHBC' in methods:
        global_partition, global_comm_sizes = get_network_partition(g, seed=COMMUNITY_SEED)

    results = {}

    for method_name in methods:
        print(f'    Evaluating: {display_method_name(method_name)}')

        if precomputed and method_name in precomputed and precomputed[method_name]:
            scores = precomputed[method_name]
        else:
            scores = get_node_scores(
                method_name, g,
                partition=global_partition,
                comm_size_map=global_comm_sizes,
            )

        ranked_nodes = get_standardized_ranked_nodes(scores)
        seeds = ranked_nodes[:top_k]

        block_curves = []
        for b in tqdm(range(num_blocks), desc=f'    {display_method_name(method_name)}', leave=False):
            repeat_curves = []
            for r in range(repeats_per_block):
                curve = run_sir_temporal(
                    g, seeds, beta, gamma,
                    max_steps=max_steps,
                    rng_seed=seed_matrix[(b, r)],
                )
                repeat_curves.append(np.asarray(curve, dtype=float))

            # One inferential observation = mean of 20 SIR realizations in one block.
            block_curve = np.mean(repeat_curves, axis=0) / n_nodes * 100.0
            block_curves.append(block_curve)

        block_curves = np.asarray(block_curves, dtype=float)  # [num_blocks, max_steps]
        mean_curve = np.mean(block_curves, axis=0)
        std_curve = np.std(block_curves, axis=0, ddof=1)
        ci95_curve = t.ppf(0.975, df=num_blocks - 1) * std_curve / np.sqrt(num_blocks)

        results[method_name] = {
            'mean': mean_curve,
            'ci95': ci95_curve,
            'raw_blocks': block_curves,
        }

    final_stats, final_pairs = compute_final_statistics(results, methods)
    return results, max_steps, top_k, final_stats, final_pairs


def compute_final_statistics(temporal_results, methods):
    """
    Compare MSH with the strongest baseline at the final time step only.

    Returns
    -------
    final_stats : dict
        Compact numeric statistics for the manuscript/global summary.
    final_pairs : DataFrame
        The 50 paired block-level final values used directly in the Wilcoxon test
        and rank-biserial effect-size calculation.
    """
    if TARGET_METHOD not in temporal_results:
        return {}, pd.DataFrame()

    target_mean = float(temporal_results[TARGET_METHOD]['mean'][-1])
    target_ci = float(temporal_results[TARGET_METHOD]['ci95'][-1])

    best_name = None
    best_mean = -np.inf
    for m in methods:
        if m == TARGET_METHOD or m not in temporal_results:
            continue
        m_final = float(temporal_results[m]['mean'][-1])
        if m_final > best_mean:
            best_mean = m_final
            best_name = m

    if best_name is None:
        return {}, pd.DataFrame()

    best_ci = float(temporal_results[best_name]['ci95'][-1])
    target_blocks = np.asarray(temporal_results[TARGET_METHOD]['raw_blocks'][:, -1], dtype=float)
    best_blocks = np.asarray(temporal_results[best_name]['raw_blocks'][:, -1], dtype=float)
    diff_blocks = target_blocks - best_blocks

    try:
        _, p_raw = wilcoxon(target_blocks, best_blocks, alternative='two-sided')
        p_raw = max(float(p_raw), 1e-20)
    except ValueError:
        p_raw = 1.0

    r_rb = float(rank_biserial_paired(target_blocks, best_blocks))

    final_stats = {
        'Best_Baseline': display_method_name(best_name),
        'MSH_Final_Mean_pct': target_mean,
        'MSH_95CI_HalfWidth': target_ci,
        'Best_Final_Mean_pct': best_mean,
        'Best_95CI_HalfWidth': best_ci,
        'Delta_F_pp': target_mean - best_mean,
        'P_Value_Raw': p_raw,
        'Rank_Biserial_r_rb': r_rb,
    }

    final_pairs = pd.DataFrame({
        'Block': np.arange(1, len(target_blocks) + 1),
        'MSH_Final_pct': target_blocks,
        f'{display_method_name(best_name)}_Final_pct': best_blocks,
        'Difference_MSH_minus_Best_pp': diff_blocks,
    })

    return final_stats, final_pairs


# ==========================================
# 4. Plotting
# ==========================================
def plot_temporal_sir(net, temporal_results, max_steps, methods, colors, markers,
                      output_dir, keep_zoom_inset=True):
    """
    Draw temporal SIR mean trajectories only.

    Confidence intervals are retained in exported data but intentionally not drawn here;
    formal paired statistics are reported only for the final time point.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    time_steps = np.arange(max_steps)

    for m in methods:
        if m not in temporal_results:
            continue

        mean_curve = temporal_results[m]['mean']
        lw = 1.6 if m == TARGET_METHOD else 1.2
        z_order = 10 if m == TARGET_METHOD else 5
        ms = 4.5 if m == TARGET_METHOD else 3.8
        alpha_val = 0.95 if m == TARGET_METHOD else 0.88
        markevery = max(3, max_steps // 10)

        ax.plot(
            time_steps, mean_curve,
            color=colors.get(m, '#000000'),
            linewidth=lw,
            linestyle='--',
            marker=markers.get(m, 'o'),
            markersize=ms,
            markerfacecolor=colors.get(m, '#000000'),
            markeredgewidth=0.5,
            markeredgecolor='black',
            markevery=markevery,
            zorder=z_order,
            alpha=alpha_val,
            label=display_method_name(m),
        )

    ax.set_title(format_network_name(net), fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel('$F(t)$ (%)', fontsize=11)
    ax.set_xlim(-1, max_steps + 1)

    all_values = [
        val
        for m in methods if m in temporal_results
        for val in temporal_results[m]['mean']
    ]
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_range = max(y_max - y_min, 1e-6)
        ax.set_ylim(max(0, y_min - y_range * 0.08), min(100, y_max + y_range * 0.08))

    if keep_zoom_inset:
        add_zoom_inset(ax, time_steps, temporal_results, methods, colors, max_steps)

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)

    plt.tight_layout(pad=0.2)

    pdf_path = os.path.join(output_dir, f'Temporal_SIR_{net}.pdf')
    png_path = os.path.join(output_dir, f'Temporal_SIR_{net}.png')
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'    [Output] Temporal figure saved: {pdf_path}')


def add_zoom_inset(ax, time_steps, temporal_results, methods, colors, max_steps):
    """Add the local zoom inset used by the temporal SIR figure."""
    mean_curves = {m: temporal_results[m]['mean'] for m in methods if m in temporal_results}

    std_per_step = []
    for tt in range(max_steps):
        values_at_t = [curve[tt] for curve in mean_curves.values()]
        std_per_step.append(np.std(values_at_t))

    window_size = max(10, min(20, int(max_steps * 0.25)))
    best_start = 0
    if max_steps > window_size:
        min_std_sum = float('inf')
        for start in range(max_steps - window_size):
            window_std_sum = sum(std_per_step[start:start + window_size])
            if window_std_sum < min_std_sum:
                min_std_sum = window_std_sum
                best_start = start

    zoom_start = best_start
    zoom_end = min(max_steps, best_start + window_size)
    axins = ax.inset_axes([0.39, 0.25, 0.36, 0.33])

    for m, mean_curve in mean_curves.items():
        lw = 1.3 if m == TARGET_METHOD else 0.9
        z_order = 10 if m == TARGET_METHOD else 5
        alpha_val = 0.95 if m == TARGET_METHOD else 0.88
        axins.plot(
            time_steps[zoom_start:zoom_end],
            mean_curve[zoom_start:zoom_end],
            color=colors.get(m, '#000000'),
            linewidth=lw,
            linestyle='--',
            zorder=z_order,
            alpha=alpha_val,
        )

    axins.set_xlim(zoom_start - 0.5, zoom_end + 0.5)

    zoom_values = [
        curve[tt]
        for curve in mean_curves.values()
        for tt in range(zoom_start, zoom_end)
    ]
    if zoom_values:
        zoom_y_min, zoom_y_max = min(zoom_values), max(zoom_values)
        zoom_y_range = zoom_y_max - zoom_y_min
        margin = 0.05 if zoom_y_range < 0.5 else zoom_y_range * 0.10
        axins.set_ylim(max(0, zoom_y_min - margin), min(100, zoom_y_max + margin))

    axins.tick_params(labelsize=7, direction='out', length=2.0, width=0.6)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#000000')

    from matplotlib.patches import Rectangle, ConnectionPatch
    zoom_y0, zoom_y1 = axins.get_ylim()
    rect = Rectangle(
        (zoom_start, zoom_y0),
        zoom_end - zoom_start,
        zoom_y1 - zoom_y0,
        fill=False,
        edgecolor='gray',
        linewidth=0.9,
        linestyle='--',
        alpha=0.85,
        transform=ax.transData,
        zorder=30,
        clip_on=False,
    )
    ax.add_patch(rect)

    con1 = ConnectionPatch(
        xyA=(zoom_end, zoom_y0), coordsA=ax.transData,
        xyB=(0, 0), coordsB=axins.transAxes,
        linestyle='--', linewidth=0.7, color='gray', alpha=0.65,
        zorder=30, clip_on=False,
    )
    con2 = ConnectionPatch(
        xyA=(zoom_end, zoom_y1), coordsA=ax.transData,
        xyB=(0, 1), coordsB=axins.transAxes,
        linestyle='--', linewidth=0.7, color='gray', alpha=0.65,
        zorder=30, clip_on=False,
    )
    ax.add_artist(con1)
    ax.add_artist(con2)


# ==========================================
# 5. Data export
# ==========================================
def save_temporal_results_to_excel(all_results, methods, output_dir):
    """
    Export only the data needed for the manuscript and reproducibility.

    Removed from the previous version:
    - temporal standard-deviation columns;
    - p-value time series;
    - paired dz;
    - block win rate;
    - redundant significance Boolean flag.
    """
    print('\n  [Export] Saving Temporal SIR results to Excel...')

    summary_records = []

    for net, data in all_results.items():
        temporal_results = data['temporal_results']
        max_steps = data['max_steps']
        final_stats = data['final_stats']
        final_pairs = data['final_pairs']

        temporal_data = {'Time_Step': np.arange(max_steps)}
        for m in methods:
            if m in temporal_results:
                label = display_method_name(m)
                temporal_data[f'{label}_Mean_pct'] = temporal_results[m]['mean']
                temporal_data[f'{label}_95CI_HalfWidth'] = temporal_results[m]['ci95']

        df_temporal = pd.DataFrame(temporal_data)

        excel_path = os.path.join(output_dir, f'Temporal_Data_{net}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_temporal.to_excel(writer, sheet_name='Temporal_Mean_CI', index=False)
            final_pairs.to_excel(writer, sheet_name='Final_Block_Pairs', index=False)

        record = {'Network': format_network_name(net)}
        record.update(final_stats)
        summary_records.append(record)
        print(f'    Saved: {excel_path}')

    if not summary_records:
        return

    summary_df = pd.DataFrame(summary_records)
    summary_df['P_Value_BH_Adjusted'] = benjamini_hochberg(summary_df['P_Value_Raw'].to_numpy())

    # Keep the numerical sheet compact and machine-readable.
    numeric_cols = [
        'Network',
        'Best_Baseline',
        'MSH_Final_Mean_pct',
        'MSH_95CI_HalfWidth',
        'Best_Final_Mean_pct',
        'Best_95CI_HalfWidth',
        'Delta_F_pp',
        'P_Value_Raw',
        'P_Value_BH_Adjusted',
        'Rank_Biserial_r_rb',
    ]
    summary_df = summary_df[numeric_cols]

    # A compact manuscript-ready view corresponding to Table 6.
    manuscript_df = pd.DataFrame({
        'Network': summary_df['Network'],
        'Best baseline': summary_df['Best_Baseline'],
        'MSH F(tc) (%)': summary_df.apply(
            lambda r: f"{r['MSH_Final_Mean_pct']:.2f} ± {r['MSH_95CI_HalfWidth']:.2f}", axis=1
        ),
        'Best F(tc) (%)': summary_df.apply(
            lambda r: f"{r['Best_Final_Mean_pct']:.2f} ± {r['Best_95CI_HalfWidth']:.2f}", axis=1
        ),
        'Delta F (pp)': summary_df['Delta_F_pp'].map(lambda x: f'{x:.2f}'),
        'BH-adjusted p': summary_df['P_Value_BH_Adjusted'].map(format_p_value),
        'r_rb': summary_df['Rank_Biserial_r_rb'].map(lambda x: f'{x:.3f}'),
    })

    summary_path = os.path.join(output_dir, 'Temporal_Final_Statistics.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Final_Statistics', index=False)
        manuscript_df.to_excel(writer, sheet_name='Manuscript_Table', index=False)

    print(f'    Saved final statistical summary: {summary_path}')


# ==========================================
# 6. Main workflow
# ==========================================
def main():
    print('=' * 64)
    print(' Experiment: Temporal SIR Propagation (Reviewer-Response Revision)')
    print('=' * 64)

    set_seed(42)

    output_dir = 'results/exp_temporal_sir_revised_stats'
    os.makedirs(output_dir, exist_ok=True)

    networks = get_network_list()

    # Internal key HOSH is preserved for compatibility; exported labels use MSH.
    methods = [
        'HOSH', 'VoteRank', 'SNIM', 'CHBC', 'ISH', 'DC',
        'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC'
    ]

    colors = {
        'HOSH': '#D63230',
        'VoteRank': '#2CA02C',
        'SNIM': '#7F7F7F',
        'CHBC': '#5D3FD3',
        'ISH': '#F08C3D',
        'DC': '#E5B25D',
        'BC': '#4FA3D1',
        'CC': '#4364B8',
        'K-Shell': '#A855A8',
        'SH': '#E2739F',
        'CI': '#8D6E63',
        'SNC': '#4DB6AC',
    }

    markers = {
        'HOSH': 'o',
        'VoteRank': 'o',
        'SNIM': 'p',
        'CHBC': '*',
        'ISH': 's',
        'DC': '^',
        'BC': 'D',
        'CC': 'X',
        'K-Shell': 'P',
        'SH': 'v',
        'CI': 'h',
        'SNC': 'H',
    }

    export_standalone_legend(methods, colors, markers, output_dir)

    # Main-paper temporal setting: fixed Top-10 seed budget.
    fixed_top_k = 10
    all_results = {}

    for i, net in enumerate(networks, 1):
        print(f'\n[{i}/{len(networks)}] Processing network: {net}')
        print('-' * 64)

        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f'  [Skip] Network {net} is empty or failed to load')
                continue

            print(f'  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}')
            top_k = min(fixed_top_k, g.number_of_nodes())
            print(f'  Seed budget: fixed Top-{top_k}')

            # Preserve the previous deterministic temporal random-seed schedule.
            temporal_base_seed = stable_int_hash('temporal_sir', net, top_k, 2026)

            temporal_results, max_steps, actual_k, final_stats, final_pairs = exp_temporal_sir(
                methods,
                g,
                top_k=top_k,
                network_name=net,
                num_blocks=NUM_BLOCKS,
                repeats_per_block=REPEATS_PER_BLOCK,
                base_seed=temporal_base_seed,
            )

            plot_temporal_sir(
                net,
                temporal_results,
                max_steps,
                methods,
                colors,
                markers,
                output_dir,
                keep_zoom_inset=True,
            )

            all_results[net] = {
                'temporal_results': temporal_results,
                'max_steps': max_steps,
                'top_k': actual_k,
                'final_stats': final_stats,
                'final_pairs': final_pairs,
            }

        except Exception as exc:
            print(f'  [Error] Failed to process {net}: {exc}')
            import traceback
            traceback.print_exc()

    if all_results:
        save_temporal_results_to_excel(all_results, methods, output_dir)

    print('\n' + '=' * 64)
    print(' Temporal SIR experiment completed.')
    print('=' * 64)
    print(f'Results saved to: {output_dir}/')


if __name__ == '__main__':
    main()
