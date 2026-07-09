"""
Experiment 6: Temporal SIR propagation experiment (revised template)

Purpose:
- Select Top-K nodes as initial infected seeds.
- Track the temporal evolution of the cumulative infection scale F(t).
- Use the same statistical protocol as the revised SIR influence-maximization experiment:
  1) 50 independent simulation blocks;
  2) 20 SIR realizations per block;
  3) shared random seeds across methods for paired comparison;
  4) 95% confidence intervals based on Student's t distribution;
  5) paired Wilcoxon test comparing HOSH with the best baseline at the final time step.

Figure policy:
- The main temporal figure only shows mean trajectories, without shaded CI/error bands.
- Statistical evidence is reported after the trajectory figures as a separate final-time
  statistical-comparison table. This keeps the temporal curves visually clean while
  still addressing uncertainty and formal paired tests.
"""

import os
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import t, wilcoxon

from hosh_methods import get_node_scores
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes


# ==========================================
# 0. Plot configuration
# ==========================================
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


def format_network_name(name):
    """Format network name consistently in figure titles."""
    s = str(name)
    return s[:1].upper() + s[1:].lower() if s else s


# ==========================================
# 1. Utility functions
# ==========================================
def export_standalone_legend(methods, colors, markers, output_dir):
    """Export a standalone legend to avoid overcrowding small temporal panels."""
    fig, ax = plt.subplots(figsize=(8.2, 1.2))
    ax.axis('off')

    handles = []
    for m in methods:
        line, = ax.plot(
            [], [], label=m,
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

    pdf_path = os.path.join(output_dir, "Temporal_Legend_Standalone.pdf")
    plt.savefig(pdf_path, format='pdf')
    plt.close()


def make_seed_matrix(num_blocks=50, repeats_per_block=20, base_seed=2026):
    """
    Generate shared random seeds for common-random-number SIR evaluation.

    The same (block, repeat) seed is used by every method under the same network
    and parameter setting, enabling paired block-level comparisons.
    """
    return {
        (b, r): base_seed + b * 1000 + r
        for b in range(num_blocks)
        for r in range(repeats_per_block)
    }


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction without extra dependencies."""
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
    ranked_q = np.clip(ranked_q, 0, 1)

    q_valid = np.empty_like(p)
    q_valid[order] = ranked_q
    q_values[valid] = q_valid
    return q_values


def fmt_mean_ci(mean, ci):
    """Format mean ± 95% CI for compact reporting."""
    return f"{mean:.2f} ± {ci:.2f}"


# ==========================================
# 2. Temporal SIR model
# ==========================================
def run_sir_temporal(graph, seeds, beta, gamma, max_steps=100, rng_seed=None):
    """
    Run one discrete-time SIR simulation and record cumulative infection size.

    Update convention:
    1. Each infected node attempts to infect each susceptible neighbor with probability beta.
    2. After infection attempts, each infected node recovers with probability gamma.
    3. Newly infected nodes become active in the next time step.
    4. F(t) records infected + recovered nodes at the beginning of each time step.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random

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

    # Defensive padding in case the loop exits unexpectedly.
    if len(temporal_infection) < max_steps:
        last_val = temporal_infection[-1] if temporal_infection else 0
        temporal_infection.extend([last_val] * (max_steps - len(temporal_infection)))

    return temporal_infection[:max_steps]


def exp_temporal_sir(methods, g, top_k=10, network_name=None,
                     num_blocks=50, repeats_per_block=20, base_seed=2026):
    """Temporal SIR propagation experiment with block-level statistics."""
    print("  [Exp: Temporal SIR] Running propagation analysis...")

    N = g.number_of_nodes()
    degrees = [d for _, d in g.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d ** 2 for d in degrees])

    gamma = 1
    threshold_multiplier = 2.5
    beta_th = gamma * (k_mean / (k2_mean - k_mean)) if (k2_mean - k_mean) > 0 else 0.0
    beta = threshold_multiplier * beta_th

    top_k = min(top_k, N)

    if N > 3000:
        max_steps = 100
    else:
        max_steps = 50

    print(f"    Graph Properties: <k>={k_mean:.2f}, <k^2>={k2_mean:.2f}")
    print(f"    Epidemic Threshold used in simulation: beta_th={beta_th:.6f}")
    print(f"    SIR Parameters: beta={beta:.6f} ({threshold_multiplier:g}×beta_th), gamma={gamma:.2f}")
    print(f"    Seed Count: {top_k} nodes ({top_k / N * 100:.2f}% of network)")
    print(f"    Simulation: {num_blocks} blocks × {repeats_per_block} repeats, {max_steps} time steps")

    seed_matrix = make_seed_matrix(num_blocks, repeats_per_block, base_seed)
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    results = {}

    for method_name in methods:
        print(f"    Evaluating: {method_name}")

        if precomputed and method_name in precomputed and precomputed[method_name]:
            scores = precomputed[method_name]
        else:
            scores = get_node_scores(method_name, g)

        ranked_nodes = get_standardized_ranked_nodes(scores)
        seeds = ranked_nodes[:top_k]

        block_curves = []
        for b in tqdm(range(num_blocks), desc=f"    {method_name}", leave=False):
            repeat_curves = []
            for r in range(repeats_per_block):
                curve = run_sir_temporal(
                    g, seeds, beta, gamma,
                    max_steps=max_steps,
                    rng_seed=seed_matrix[(b, r)],
                )
                repeat_curves.append(np.asarray(curve, dtype=float))

            block_curve = np.mean(repeat_curves, axis=0) / N * 100.0
            block_curves.append(block_curve)

        block_curves = np.asarray(block_curves)  # shape: [num_blocks, max_steps]
        mean_curve = np.mean(block_curves, axis=0)
        std_curve = np.std(block_curves, axis=0, ddof=1)
        ci95_curve = t.ppf(0.975, df=num_blocks - 1) * std_curve / np.sqrt(num_blocks)

        results[method_name] = {
            'mean': mean_curve,
            'std': std_curve,
            'ci95': ci95_curve,
            'raw_blocks': block_curves,
            'seeds': seeds,
        }

    final_stats, pvalue_series = compute_temporal_statistics(results, methods, max_steps)
    return results, max_steps, top_k, final_stats, pvalue_series


def compute_temporal_statistics(temporal_results, methods, max_steps):
    """
    Compare HOSH with the strongest non-HOSH baseline at the final time step.

    Statistical protocol:
    - block-level paired comparison under shared SIR random seeds;
    - paired Wilcoxon signed-rank test;
    - paired mean difference and paired effect size;
    - win rate across simulation blocks;
    - optional p-value time series retained for supplementary checking.
    """
    if 'HOSH' not in temporal_results:
        return {}, pd.DataFrame()

    hosh_final = temporal_results['HOSH']['mean'][-1]
    hosh_ci = temporal_results['HOSH']['ci95'][-1]
    best_name = None
    best_final = -np.inf

    for m in methods:
        if m == 'HOSH' or m not in temporal_results:
            continue
        m_final = temporal_results[m]['mean'][-1]
        if m_final > best_final:
            best_final = m_final
            best_name = m

    best_ci = np.nan
    p_val = np.nan
    delta_blocks_mean = np.nan
    paired_effect_dz = np.nan
    win_rate = np.nan

    if best_name is not None:
        best_ci = temporal_results[best_name]['ci95'][-1]
        hosh_blocks = temporal_results['HOSH']['raw_blocks'][:, -1]
        best_blocks = temporal_results[best_name]['raw_blocks'][:, -1]
        diff_blocks = hosh_blocks - best_blocks
        delta_blocks_mean = float(np.mean(diff_blocks))
        diff_std = float(np.std(diff_blocks, ddof=1))
        paired_effect_dz = delta_blocks_mean / diff_std if diff_std > 0 else np.nan
        win_rate = float(np.mean(diff_blocks > 0) * 100.0)
        try:
            _, p_val = wilcoxon(hosh_blocks, best_blocks, alternative='two-sided')
            p_val = max(float(p_val), 1e-20)
        except ValueError:
            p_val = 1.0

    final_stats = {
        'Best_Baseline_Final': best_name,
        'HOSH_Final(%)': hosh_final,
        'HOSH_95%_CI': hosh_ci,
        'HOSH_Final_Mean±CI': fmt_mean_ci(hosh_final, hosh_ci),
        'Best_Final(%)': best_final,
        'Best_95%_CI': best_ci,
        'Best_Final_Mean±CI': fmt_mean_ci(best_final, best_ci) if not np.isnan(best_ci) else '',
        'Delta_Final(%)': hosh_final - best_final,
        'Mean_Paired_Difference(%)': delta_blocks_mean,
        'Paired_Effect_dz': paired_effect_dz,
        'Win_Rate_Blocks(%)': win_rate,
        'P_Value_Final_vs_Best': p_val,
    }

    # Optional p-value series: HOSH vs the strongest baseline at each time step.
    # This is exported for supplementary checking, but the main figure remains uncluttered.
    p_records = []
    for tt in range(max_steps):
        best_t_name = None
        best_t_mean = -np.inf
        for m in methods:
            if m == 'HOSH' or m not in temporal_results:
                continue
            m_mean = temporal_results[m]['mean'][tt]
            if m_mean > best_t_mean:
                best_t_mean = m_mean
                best_t_name = m

        p_val_t = np.nan
        delta_t = np.nan
        if best_t_name is not None:
            hosh_t = temporal_results['HOSH']['raw_blocks'][:, tt]
            best_t = temporal_results[best_t_name]['raw_blocks'][:, tt]
            delta_t = temporal_results['HOSH']['mean'][tt] - best_t_mean
            try:
                _, p_val_t = wilcoxon(hosh_t, best_t, alternative='two-sided')
                p_val_t = max(float(p_val_t), 1e-20)
            except ValueError:
                p_val_t = 1.0

        p_records.append({
            'Time_Step': tt,
            'Best_Baseline': best_t_name,
            'HOSH_Mean(%)': temporal_results['HOSH']['mean'][tt],
            'Best_Mean(%)': best_t_mean,
            'Delta_vs_Best(%)': delta_t,
            'P_Value_vs_Best': p_val_t,
        })

    return final_stats, pd.DataFrame(p_records)

# ==========================================
# 3. Plotting
# ==========================================
def plot_temporal_sir(net, temporal_results, max_steps, methods, colors, markers, output_dir, top_k,
                      show_ci_for=(), keep_zoom_inset=True):
    """
    Draw temporal SIR curves.

    Design choice:
    - Show mean trajectories only.
    - Do not draw shaded CI/error bands in the temporal trajectory panel.
    - Report final-time statistical tests separately.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    time_steps = np.arange(max_steps)

    for m in methods:
        if m not in temporal_results:
            continue

        mean_curve = temporal_results[m]['mean']
        ci_curve = temporal_results[m]['ci95']

        lw = 1.6 if m == 'HOSH' else 1.2
        z_order = 10 if m == 'HOSH' else 5
        ms = 4.5 if m == 'HOSH' else 3.8
        alpha_val = 0.95 if m == 'HOSH' else 0.88
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
            label=m,
        )

        # Optional CI bands are disabled by default for a cleaner main figure.
        if m in show_ci_for:
            ax.fill_between(
                time_steps,
                np.maximum(0, mean_curve - ci_curve),
                np.minimum(100, mean_curve + ci_curve),
                color=colors.get(m, '#000000'),
                alpha=0.12,
                linewidth=0,
                zorder=max(1, z_order - 2),
            )

    ax.set_title(format_network_name(net), fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel("$F(t)$ (%)", fontsize=11)
    ax.set_xlim(-1, max_steps + 1)

    all_values = [val for m in methods if m in temporal_results for val in temporal_results[m]['mean']]
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

    pdf_path = os.path.join(output_dir, f"Temporal_SIR_{net}.pdf")
    png_path = os.path.join(output_dir, f"Temporal_SIR_{net}.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"    [Output] Temporal figure saved: {pdf_path}")


def add_zoom_inset(ax, time_steps, temporal_results, methods, colors, max_steps):
    """Add the local zoom inset used by the temporal SIR figure, without CI shading."""
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
        lw = 1.3 if m == 'HOSH' else 0.9
        z_order = 10 if m == 'HOSH' else 5
        alpha_val = 0.95 if m == 'HOSH' else 0.88
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

    zoom_values = [curve[tt] for curve in mean_curves.values() for tt in range(zoom_start, zoom_end)]
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
        fill=False, edgecolor='gray', linewidth=0.9,
        linestyle='--', alpha=0.85, transform=ax.transData,
        zorder=30, clip_on=False,
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

def plot_temporal_pvalues(net, pvalue_series, output_dir):
    """Save p-values as a separate small figure, rather than as an inset."""
    if pvalue_series is None or pvalue_series.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    x = pvalue_series['Time_Step'].to_numpy()
    y = pvalue_series['P_Value_vs_Best'].to_numpy(dtype=float)

    ax.plot(x, y, marker='^', color='indigo', markersize=2.8, linewidth=1.0)
    ax.set_yscale('log')
    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8)
    ax.set_title(format_network_name(net), fontsize=11, fontweight='bold', pad=5)
    ax.set_xlabel('$t$')
    ax.set_ylabel('P-value')
    ax.set_xlim(x.min(), x.max())

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)

    plt.tight_layout(pad=0.2)
    pdf_path = os.path.join(output_dir, f"Temporal_PValue_{net}.pdf")
    plt.savefig(pdf_path, format='pdf')
    plt.close()


def plot_final_statistical_summary(summary_df, output_dir):
    """Export a compact PDF table for final-time statistical comparisons."""
    if summary_df is None or summary_df.empty:
        return

    display_cols = [
        'Network', 'Best_Baseline_Final', 'HOSH_Final_Mean±CI', 'Best_Final_Mean±CI',
        'Delta_Final(%)', 'P_Value_Final_vs_Best', 'Q_Value_BH', 'Paired_Effect_dz',
        'Win_Rate_Blocks(%)'
    ]
    table_df = summary_df[display_cols].copy()
    rename = {
        'Best_Baseline_Final': 'Best baseline',
        'HOSH_Final_Mean±CI': 'HOSH final (%)',
        'Best_Final_Mean±CI': 'Best final (%)',
        'Delta_Final(%)': 'ΔF (%)',
        'P_Value_Final_vs_Best': 'p',
        'Q_Value_BH': 'q',
        'Paired_Effect_dz': 'dz',
        'Win_Rate_Blocks(%)': 'Win (%)',
    }
    table_df = table_df.rename(columns=rename)

    for col in ['ΔF (%)', 'dz', 'Win (%)']:
        if col in table_df:
            table_df[col] = table_df[col].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    for col in ['p', 'q']:
        if col in table_df:
            table_df[col] = table_df[col].map(lambda x: '' if pd.isna(x) else (f'{x:.2e}' if x < 0.001 else f'{x:.3f}'))

    fig_height = max(1.8, 0.34 * len(table_df) + 0.7)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis('off')
    tab = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1, 1.15)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAEAEA')
    plt.tight_layout(pad=0.2)
    out_pdf = os.path.join(output_dir, 'Temporal_Final_Statistical_Test_Table.pdf')
    plt.savefig(out_pdf, format='pdf')
    plt.close()


# ==========================================
# 4. Data export
# ==========================================
def save_temporal_results_to_excel(all_results, methods, output_dir):
    """Save temporal SIR mean/std/CI and final-time statistical comparisons to Excel."""
    print("\n  [Export] Saving Temporal SIR results to Excel...")

    summary_records = []

    for net, data in all_results.items():
        temporal_results = data['temporal_results']
        max_steps = data['max_steps']
        final_stats = data['final_stats']
        pvalue_series = data['pvalue_series']

        df_data = {'Time_Step': np.arange(max_steps)}
        for m in methods:
            if m in temporal_results:
                df_data[f'{m}_Mean(%)'] = temporal_results[m]['mean']
                df_data[f'{m}_Std'] = temporal_results[m]['std']
                df_data[f'{m}_95%_CI'] = temporal_results[m]['ci95']

        df_temporal = pd.DataFrame(df_data)
        df_final_stats = pd.DataFrame([final_stats])

        excel_path = os.path.join(output_dir, f"Temporal_Data_{net}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_temporal.to_excel(writer, sheet_name='Temporal_Mean_CI', index=False)
            df_final_stats.to_excel(writer, sheet_name='Final_Comparison', index=False)
            if pvalue_series is not None and not pvalue_series.empty:
                pvalue_series.to_excel(writer, sheet_name='PValue_TimeSeries', index=False)

        record = {'Network': net}
        record.update(final_stats)
        summary_records.append(record)
        print(f"    Saved: {excel_path}")

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df['Q_Value_BH'] = benjamini_hochberg(summary_df['P_Value_Final_vs_Best'].to_numpy())
        summary_df['Significant_0.05_BH'] = summary_df['Q_Value_BH'] < 0.05

        summary_path = os.path.join(output_dir, "Temporal_Final_Statistical_Test_Summary.xlsx")
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Final_Statistical_Tests', index=False)
        print(f"    Saved statistical summary: {summary_path}")
        plot_final_statistical_summary(summary_df, output_dir)

# ==========================================
# 5. Main workflow
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Temporal SIR Propagation (Revised)")
    print("=" * 60)

    set_seed(42)

    output_dir = "results/exp_temporal_sir"
    os.makedirs(output_dir, exist_ok=True)

    networks = get_network_list()

    # Keep the baseline set consistent with the revised influence-maximization experiment.
    methods = ['HOSH', 'VoteRank', 'SNIM', 'CHBC', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']

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

    # Default: keep the original fixed Top-10 temporal setting.
    # For a supplementary scale-normalized temporal test, set seed_budget_mode='ratio'.
    seed_budget_mode = 'fixed'   # options: 'fixed' or 'ratio'
    fixed_top_k = 10
    seed_ratio = 0.01            # used only when seed_budget_mode == 'ratio'

    all_results = {}

    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")
        print("-" * 60)

        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f"  [Skip] Network {net} is empty or failed to load")
                continue

            print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

            if seed_budget_mode == 'ratio':
                top_k = max(1, int(g.number_of_nodes() * seed_ratio))
                print(f"  Seed budget mode: ratio={seed_ratio:.2%}; top_k={top_k}")
            else:
                top_k = fixed_top_k
                print(f"  Seed budget mode: fixed Top-{top_k}")

            temporal_base_seed = stable_int_hash('temporal_sir', net, top_k, 2026)
            temporal_results, max_steps, actual_k, final_stats, pvalue_series = exp_temporal_sir(
                methods, g, top_k=top_k, network_name=net,
                num_blocks=50, repeats_per_block=20, base_seed=temporal_base_seed,
            )

            plot_temporal_sir(
                net, temporal_results, max_steps, methods,
                colors, markers, output_dir, actual_k,
                show_ci_for=(),
                keep_zoom_inset=True,
            )
            # P-value time series is exported to Excel; the main paper uses the final-time
            # statistical-test summary table instead of another inset/curve.

            all_results[net] = {
                'temporal_results': temporal_results,
                'max_steps': max_steps,
                'final_stats': final_stats,
                'pvalue_series': pvalue_series,
            }

        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        save_temporal_results_to_excel(all_results, methods, output_dir)

    print("\n" + "=" * 60)
    print(" Experiment: Temporal SIR Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
