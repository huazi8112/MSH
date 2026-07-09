"""
实验1: SIR影响力最大化实验（逐网络输出图 + BH调整p值）
包含：95%置信区间误差棒、MSH/HOSH vs strongest baseline 的配对 Wilcoxon 检验、
逐网络 Benjamini-Hochberg 调整后的 p-value 小图、全数据导出。

说明：
- 每完成一个网络，立即输出该网络的 PDF/PNG 图和 Excel 数据。
- 小图纵轴仍显示为 P-value；图注中说明这些 p-values 经过 BH 调整。
- BH 调整在每个网络内部对 10 个 seed ratios 的比较进行。
- Excel 同时保存 raw p-value 和 BH-adjusted p-value。
- 代码内部仍沿用原方法名 HOSH；论文正文可统一写为 MSH。
"""
import networkx as nx
import numpy as np
import random
import hashlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon, t

from hosh_methods import get_node_scores, get_network_partition
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes

# ==========================================
# 0. 基础绘图配置
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
    'axes.spines.top': True,
    'axes.spines.right': True,
})


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def stable_int_hash(*items, modulo=2**32 - 1):
    """Stable integer hash for deterministic seed derivation across Python processes."""
    text = "||".join(str(x) for x in items)
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def get_shared_sir_seed(network_name, ratio_idx, sample_idx, repeat_idx, base_seed=20260710):
    """
    Method-independent shared SIR seed.
    The same network, seed ratio, block, and repeat use the same seed for all methods.
    Uses stable hashing instead of Python's randomized hash().
    """
    return stable_int_hash(base_seed, network_name, ratio_idx, sample_idx, repeat_idx)


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([])

    order = np.argsort(p)
    ranked_p = p[order]
    q_ranked = ranked_p * n / (np.arange(1, n + 1))

    # Enforce monotonicity from largest p to smallest p.
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q


# ==========================================
# 1. 独立图例生成函数
# ==========================================
def export_standalone_legend(methods, colors, markers, output_dir):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis('off')

    dummy_lines = []
    for m in methods:
        line, = ax.plot([], [], label=m, color=colors.get(m, '#000000'),
                        marker=markers.get(m, "o"), linestyle='--',
                        linewidth=1.5, markersize=5,
                        markerfacecolor=colors.get(m, '#000000'),
                        markeredgecolor='black', markeredgewidth=0.5)
        dummy_lines.append(line)

    ax.legend(handles=dummy_lines, loc='center', ncol=6,
              frameon=True, fancybox=False, shadow=False, edgecolor='black', fontsize=10)

    pdf_path = os.path.join(output_dir, "Legend_Standalone.pdf")
    plt.savefig(pdf_path, format='pdf')
    plt.close()


# ==========================================
# 2. SIR 传播模型与数据收集
# ==========================================
def run_sir_simulation(graph, seeds, beta, gamma, max_steps=1000, rng_seed=None):
    """Discrete-time synchronous SIR simulation using a local RNG."""
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    infected_nodes = set(n for n in seeds if graph.has_node(n))
    recovered_nodes = set()

    if not infected_nodes:
        return 0

    for _ in range(max_steps):
        if not infected_nodes:
            break
        new_infected, new_recovered = set(), set()
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

    return len(recovered_nodes) + len(infected_nodes)


def exp_influence_maximization(methods, g, network_name=None):
    N = g.number_of_nodes()
    degrees = [d for _, d in g.degree()]
    k_mean, k2_mean = np.mean(degrees), np.mean([d ** 2 for d in degrees])

    gamma = 1
    beta_th = gamma * (k_mean / (k2_mean - k_mean)) if (k2_mean - k_mean) > 0 else 0
    beta = 2.5 * beta_th

    num_samples = 50
    repeat_times = 20
    seed_ratios = np.arange(0.01, 0.11, 0.01)
    results = {}

    precomputed = load_precomputed_rankings(network_name) if network_name else None
    global_partition, global_comm_sizes = None, None
    if 'CHBC' in methods:
        global_partition, global_comm_sizes = get_network_partition(g, seed=42)

    for method_name in methods:
        if precomputed and method_name in precomputed:
            scores = precomputed[method_name]
        else:
            scores = get_node_scores(method_name, g, partition=global_partition, comm_size_map=global_comm_sizes)

        ranked_nodes = get_standardized_ranked_nodes(scores)
        means, stds, raw_samples_list = [], [], []

        for ratio_idx, ratio in enumerate(tqdm(seed_ratios, desc=f"    {method_name}", leave=False)):
            k = max(1, int(N * ratio))
            seeds = ranked_nodes[:k]

            sample_results = []
            for sample_idx in range(num_samples):
                total_infected = 0
                for repeat_idx in range(repeat_times):
                    sim_seed = get_shared_sir_seed(network_name, ratio_idx, sample_idx, repeat_idx)
                    total_infected += run_sir_simulation(g, seeds, beta, gamma, rng_seed=sim_seed)
                sample_results.append((total_infected / repeat_times) / N * 100)

            means.append(np.mean(sample_results))
            stds.append(np.std(sample_results, ddof=1))
            raw_samples_list.append(sample_results)

        results[method_name] = {'mean': means, 'std': stds, 'raw_samples': raw_samples_list}

    return seed_ratios, results


# ==========================================
# 3. 显著性检验：HOSH/MSH vs strongest baseline
# ==========================================
def compute_vs_best_tests(x_ratios, sir_results, methods, target_method='HOSH'):
    """
    For each seed ratio, compare target_method with the strongest baseline
    using paired Wilcoxon signed-rank tests based on shared SIR blocks.
    Returns raw p-values and best baseline names.
    """
    p_values = []
    best_baselines = []

    if target_method not in sir_results:
        return [1.0] * len(x_ratios), ["None"] * len(x_ratios)

    for i in range(len(x_ratios)):
        target_samp = sir_results[target_method]['raw_samples'][i]
        best_baseline_mean, best_baseline_samp, best_name = -np.inf, None, "None"

        for m in methods:
            if m == target_method or m not in sir_results:
                continue
            if sir_results[m]['mean'][i] > best_baseline_mean:
                best_baseline_mean = sir_results[m]['mean'][i]
                best_baseline_samp = sir_results[m]['raw_samples'][i]
                best_name = m

        best_baselines.append(best_name)
        if best_baseline_samp is not None:
            try:
                target_arr = np.asarray(target_samp, dtype=float)
                best_arr = np.asarray(best_baseline_samp, dtype=float)
                _, p_val = wilcoxon(target_arr, best_arr, alternative='two-sided')
                p_values.append(max(float(p_val), 1e-20))
            except ValueError:
                p_values.append(1.0)
        else:
            p_values.append(1.0)

    return p_values, best_baselines


# ==========================================
# 4. 绘图：95%置信区间 + BH-adjusted p-value 小图
# ==========================================
def plot_sir_results(net, x_ratios, sir_results, methods, colors, markers, output_dir, adjusted_p_values=None):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    num_samples = 50

    # 绘制主折线图
    for m in methods:
        if m not in sir_results:
            continue
        mean_vals = np.array(sir_results[m]['mean'])
        std_vals = np.array(sir_results[m]['std'])

        # 使用基于 50 个 block mean 的 t 分布 95% 置信区间作为误差棒
        sem_vals = std_vals / np.sqrt(num_samples)
        ci_95_vals = t.ppf(0.975, df=num_samples - 1) * sem_vals

        lw, ms = (1.6, 4.5) if m == 'HOSH' else (1.2, 3.8)
        z_order, alpha_val = (10, 0.95) if m == 'HOSH' else (5, 0.88)

        ax.errorbar(x_ratios * 100, mean_vals, yerr=ci_95_vals,
                    fmt=f'--{markers.get(m, "o")}',
                    color=colors.get(m, '#000000'), linewidth=lw, markersize=ms,
                    markerfacecolor=colors.get(m, '#000000'),
                    markeredgecolor='black', markeredgewidth=0.5,
                    ecolor=colors.get(m, '#000000'), elinewidth=0.6, capsize=1.2,
                    alpha=alpha_val, zorder=z_order)

    # 指定小图位置，避免遮挡主曲线
    left_top_networks = ['infect', 'email', 'hamster', 'usair', 'polblogs', 'power', 'lesmis']
    if net.lower() in left_top_networks:
        inset_pos = [0.12, 0.70, 0.35, 0.22]
    else:
        inset_pos = [0.60, 0.13, 0.35, 0.22]

    # 绘制 BH-adjusted p-value 小图；纵轴标签保持为 P-value，图注中说明为 BH 调整后。
    if adjusted_p_values is not None:
        p_plot = np.asarray(adjusted_p_values, dtype=float)
        p_plot = np.maximum(p_plot, 1e-20)
        axins = ax.inset_axes(inset_pos)
        axins.plot(x_ratios * 100, p_plot, marker='^', color='indigo', markersize=2.5, linewidth=1.0)
        axins.set_yscale('log')
        axins.set_ylabel('P-value', fontsize=6, labelpad=1)
        axins.tick_params(axis='both', which='major', labelsize=5, length=1.5)
        axins.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8)
        axins.set_xlim(0.8, 10.2)
        axins.set_xticks(np.arange(1, 11))
        axins.set_xlabel('$p$ (%)', fontsize=6, labelpad=0.5)

    ax.set_title(net.capitalize(), fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('$p$ (%)', fontsize=11)
    ax.set_ylabel('$F(t_c)$ (%)', fontsize=11)
    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(np.arange(1, 11))

    plt.tight_layout(pad=0.2)
    pdf_path = os.path.join(output_dir, f"SIR_{net}_stat.pdf")
    png_path = os.path.join(output_dir, f"SIR_{net}_stat.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png')
    plt.close()


# ==========================================
# 5. 主流程
# ==========================================
def main():
    set_seed(42)
    output_dir = "results/exp_sir_influence"
    os.makedirs(output_dir, exist_ok=True)
    networks = get_network_list()
    methods = ['HOSH', 'VoteRank', 'SNIM', 'CHBC', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']

    colors = {'HOSH': '#D63230', 'ISH': '#F08C3D', 'DC': '#E5B25D', 'BC': '#4FA3D1',
              'CC': '#4364B8', 'K-Shell': '#A855A8', 'SH': '#E2739F', 'CI': '#8D6E63',
              'SNC': '#4DB6AC', 'VoteRank': '#2CA02C', 'SNIM': '#7F7F7F', 'CHBC': '#5D3FD3'}
    markers = {'HOSH': 'o', 'VoteRank': 'o', 'SNIM': 'p', 'CHBC': '*', 'ISH': 's',
               'DC': '^', 'BC': 'D', 'CC': 'X', 'K-Shell': 'P', 'SH': 'v', 'CI': 'h', 'SNC': 'H'}

    export_standalone_legend(methods, colors, markers, output_dir)

    all_test_records = []

    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")
        try:
            g = download_and_load_graph(net)
            if not g or g.number_of_nodes() == 0:
                continue

            x_ratios, sir_results = exp_influence_maximization(methods, g, net)
            raw_p_values, best_baselines = compute_vs_best_tests(
                x_ratios, sir_results, methods, target_method='HOSH'
            )

            # 为满足“每完成一个网络立即出图”，这里对该网络的 10 个 seed ratios 做 BH 调整。
            adjusted_p_values = benjamini_hochberg(raw_p_values)

            plot_sir_results(
                net, x_ratios, sir_results, methods, colors, markers, output_dir,
                adjusted_p_values=adjusted_p_values
            )
            print(f"    [Output] PDF/PNG graph saved for {net}.")

            df_data = {'Seed_Ratio_%': x_ratios * 100}
            for m in methods:
                if m in sir_results:
                    df_data[f'{m}_Mean(%)'] = sir_results[m]['mean']
                    df_data[f'{m}_Std'] = sir_results[m]['std']
                    df_data[f'{m}_95%_CI'] = t.ppf(0.975, df=49) * (
                        np.array(sir_results[m]['std']) / np.sqrt(50)
                    )

            df_data['P_Value_raw_vs_Best'] = raw_p_values
            df_data['P_Value_BH_adjusted_vs_Best'] = adjusted_p_values
            df_data['Best_Baseline'] = best_baselines

            excel_path = os.path.join(output_dir, f"SIR_AllData_{net}.xlsx")
            pd.DataFrame(df_data).to_excel(excel_path, index=False)
            print(f"    [Output] Excel Data saved for {net}: {excel_path}")

            for ratio, raw_p, adj_p, best_name in zip(
                x_ratios, raw_p_values, adjusted_p_values, best_baselines
            ):
                all_test_records.append({
                    'Network': net,
                    'Seed_Ratio_%': ratio * 100,
                    'Best_Baseline': best_name,
                    'P_Value_raw_vs_Best': raw_p,
                    'P_Value_BH_adjusted_vs_Best': adj_p,
                })

            # 每个网络完成后更新一次总统计表，便于中途查看结果。
            stat_summary = pd.DataFrame(all_test_records)
            stat_summary_path = os.path.join(output_dir, "SIR_vs_Best_Statistical_Tests_BH_Incremental.csv")
            stat_summary_xlsx_path = os.path.join(output_dir, "SIR_vs_Best_Statistical_Tests_BH_Incremental.xlsx")
            stat_summary.to_csv(stat_summary_path, index=False)
            stat_summary.to_excel(stat_summary_xlsx_path, index=False)
            print(f"    [Output] Incremental statistical summary updated.")

        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")


if __name__ == "__main__":
    main()
