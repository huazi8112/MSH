"""
实验：WCycle-style 邻域 Jaccard 结构相似性实验（逐网络输出图）

目的
----
评价不同节点排序方法所选 Top-k 节点组之间的局部结构重叠/冗余。
采用 WCycle 文献中的 neighborhood Jaccard structural similarity：

    J_ij = |Gamma(i) ∩ Gamma(j)| / |Gamma(i) ∪ Gamma(j)|

对选定的 k 个节点，计算所有无序节点对的平均值：

    J_s = 2 / [k(k-1)] * sum_{i<j} J_ij

J_s 越小，表示所选节点的邻域重叠越低、结构冗余越小。

比例规则
--------
- 默认希望考察 10 个连续整数比例。
- 若某网络在 1% 时可选节点数 k=floor(N*p) >= 2，则使用 1%-10%。
- 若低比例下 k<2，则整个比例区间自动向后平移，直到起始比例首次满足 k>=2。
  例如：
    * 若 1% 不足 2 个节点、2% 已满足，则使用 2%-11%；
    * 若 1%、2% 均不足 2 个节点、3% 已满足，则使用 3%-12%。
- 始终保留 10 个连续比例点。
- k 的计算方式与 SIR 主实验一致：k = int(N * ratio)，即向下取整。

输出
----
results/exp_jaccard_structural_similarity/
    Legend_Standalone.pdf
    Jaccard_<network>.pdf
    Jaccard_<network>.png
    Jaccard_AllData_<network>.xlsx
    Jaccard_AllNetworks_Long.xlsx
    Jaccard_AllNetworks_Wide.xlsx

说明
----
- 该实验是确定性结构诊断，不进行 Monte Carlo，因此不添加 95% CI、p-value 或误差棒。
- 代码内部仍沿用 HOSH；论文图例显示为 MSH。
- 绘图字体、颜色、marker、线宽和单网络画幅尽量与 SIR 主实验保持一致。
"""

import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hosh_methods import get_node_scores, get_network_partition
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes


# ==========================================
# 0. 基础绘图配置：与 SIR 主实验保持一致
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


MASTER_SEED = 42
N_RATIO_POINTS = 10
MIN_SELECTED_NODES = 2

METHODS = [
    'HOSH', 'VoteRank', 'SNIM', 'CHBC', 'ISH', 'DC',
    'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC'
]

DISPLAY_NAMES = {
    'HOSH': 'MSH',
    'VoteRank': 'VoteRank',
    'SNIM': 'SNIM',
    'CHBC': 'CHBC',
    'ISH': 'ISH',
    'DC': 'DC',
    'BC': 'BC',
    'CC': 'CC',
    'K-Shell': 'K-Shell',
    'SH': 'SH',
    'CI': 'CI',
    'SNC': 'SNC',
}

COLORS = {
    'HOSH': '#D63230',
    'ISH': '#F08C3D',
    'DC': '#E5B25D',
    'BC': '#4FA3D1',
    'CC': '#4364B8',
    'K-Shell': '#A855A8',
    'SH': '#E2739F',
    'CI': '#8D6E63',
    'SNC': '#4DB6AC',
    'VoteRank': '#2CA02C',
    'SNIM': '#7F7F7F',
    'CHBC': '#5D3FD3',
}

MARKERS = {
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================
# 1. 比例自动延后
# ==========================================
def get_shifted_seed_ratios(
    n_nodes: int,
    n_points: int = N_RATIO_POINTS,
    min_selected_nodes: int = MIN_SELECTED_NODES,
):
    """
    返回 10 个连续整数百分比。

    规则：
    1) 从 1% 开始检查；
    2) 找到第一个满足 int(N * p) >= 2 的整数百分比；
    3) 从该比例开始连续取 10 个点。

    例如：
        N=332 -> 1%-10%（1% 已有 3 个节点）
        N=112 -> 2%-11%（1% 只有 1 个节点）
        N=77  -> 3%-12%（1%、2% 均不足 2 个节点）
    """
    if n_nodes < min_selected_nodes:
        raise ValueError(
            f"Network has only {n_nodes} node(s), fewer than the required "
            f"{min_selected_nodes} nodes."
        )

    start_percent = None
    for p_int in range(1, 101):
        k = int(n_nodes * (p_int / 100.0))
        if k >= min_selected_nodes:
            start_percent = p_int
            break

    if start_percent is None:
        raise ValueError("Could not find a valid seed proportion with at least 2 nodes.")

    end_percent = start_percent + n_points - 1
    if end_percent > 100:
        raise ValueError(
            f"Cannot construct {n_points} consecutive integer ratios from "
            f"{start_percent}% without exceeding 100%."
        )

    percentages = np.arange(start_percent, end_percent + 1, dtype=int)
    ratios = percentages / 100.0
    return ratios, percentages


# ==========================================
# 2. 独立图例：与 SIR 主实验保持一致
# ==========================================
def export_standalone_legend(methods, colors, markers, output_dir):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis('off')

    dummy_lines = []
    for m in methods:
        lw, ms = (1.6, 5.0) if m == 'HOSH' else (1.2, 4.2)
        line, = ax.plot(
            [], [],
            label=DISPLAY_NAMES.get(m, m),
            color=colors.get(m, '#000000'),
            marker=markers.get(m, 'o'),
            linestyle='--',
            linewidth=lw,
            markersize=ms,
            markerfacecolor=colors.get(m, '#000000'),
            markeredgecolor='black',
            markeredgewidth=0.5,
        )
        dummy_lines.append(line)

    ax.legend(
        handles=dummy_lines,
        loc='center',
        ncol=6,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='black',
        fontsize=10,
    )

    pdf_path = os.path.join(output_dir, "Legend_Standalone.pdf")
    plt.savefig(pdf_path, format='pdf')
    plt.close()


# ==========================================
# 3. WCycle-style Jaccard 结构相似性
# ==========================================
def neighborhood_jaccard(graph, u, v, neighbor_cache=None):
    """
    WCycle-style pairwise neighborhood Jaccard similarity:

        J_ij = |Gamma(i) ∩ Gamma(j)| / |Gamma(i) ∪ Gamma(j)|

    Gamma(i) 为节点 i 的开放邻居集合，不包含节点自身。
    """
    if neighbor_cache is None:
        nu = set(graph.neighbors(u))
        nv = set(graph.neighbors(v))
    else:
        nu = neighbor_cache[u]
        nv = neighbor_cache[v]

    union = nu | nv
    if len(union) == 0:
        return 0.0

    return len(nu & nv) / len(union)


def average_structural_similarity(graph, selected_nodes, neighbor_cache=None):
    """
    对 Top-k 节点组中的所有无序节点对求平均 Jaccard 相似性。

    与 WCycle 文献按 i != j 的有序求和再除以 k(k-1) 数学等价：

        J_s = 2/[k(k-1)] * sum_{i<j} J_ij

    k < 2 时无定义，但本实验的比例自动延后机制保证 k >= 2。
    """
    selected_nodes = list(selected_nodes)
    k = len(selected_nodes)

    if k < 2:
        return np.nan

    total = 0.0
    pair_count = 0

    for u, v in combinations(selected_nodes, 2):
        total += neighborhood_jaccard(
            graph, u, v, neighbor_cache=neighbor_cache
        )
        pair_count += 1

    return total / pair_count if pair_count > 0 else np.nan


# ==========================================
# 4. 排名加载与结构相似性计算
# ==========================================
def load_method_ranking(method_name, graph, network_name, partition=None, comm_sizes=None):
    """
    优先读取预计算 ranking；若不存在则现场计算。
    与 SIR 主实验保持相同的 standardized deterministic tie-breaking。
    """
    precomputed = load_precomputed_rankings(network_name)

    if precomputed and method_name in precomputed and precomputed[method_name] is not None:
        scores = precomputed[method_name]
    else:
        scores = get_node_scores(
            method_name,
            graph,
            partition=partition,
            comm_size_map=comm_sizes,
        )

    return get_standardized_ranked_nodes(scores)


def exp_jaccard_structural_similarity(methods, graph, network_name):
    """
    对单个网络完成：
    - 自动确定有效的 10 个连续 seed ratios；
    - 加载各方法 ranking；
    - 计算各比例下 Top-k 节点的平均 structural similarity J_s。
    """
    n = graph.number_of_nodes()
    seed_ratios, seed_percentages = get_shifted_seed_ratios(n)

    print(
        f"    Ratio range: {seed_percentages[0]}%-{seed_percentages[-1]}% "
        f"(N={n}, minimum selected nodes={MIN_SELECTED_NODES})"
    )

    # 预缓存邻居集合，避免重复 set(graph.neighbors())。
    neighbor_cache = {
        node: set(graph.neighbors(node))
        for node in graph.nodes()
    }

    global_partition, global_comm_sizes = None, None
    if 'CHBC' in methods:
        global_partition, global_comm_sizes = get_network_partition(
            graph, seed=42
        )

    results = {}
    rankings = {}

    for method_name in methods:
        rankings[method_name] = load_method_ranking(
            method_name,
            graph,
            network_name,
            partition=global_partition,
            comm_sizes=global_comm_sizes,
        )

    for method_name in tqdm(methods, desc="    Methods", leave=False):
        ranked_nodes = rankings[method_name]

        jaccard_values = []
        selected_k_values = []
        actual_ratio_values = []

        for ratio in seed_ratios:
            # 与 SIR 主实验一致：直接向下取整，不人为 max(2, ...)
            k = int(n * ratio)

            # 理论上由于 get_shifted_seed_ratios，所有点均应满足 k>=2。
            if k < MIN_SELECTED_NODES:
                raise RuntimeError(
                    f"Unexpected k={k} for {network_name}, "
                    f"ratio={ratio:.2%}. Ratio shifting failed."
                )

            selected_nodes = ranked_nodes[:k]

            j_s = average_structural_similarity(
                graph,
                selected_nodes,
                neighbor_cache=neighbor_cache,
            )

            jaccard_values.append(j_s)
            selected_k_values.append(k)
            actual_ratio_values.append(k / n * 100.0)

        results[method_name] = {
            'Js': np.asarray(jaccard_values, dtype=float),
            'k': np.asarray(selected_k_values, dtype=int),
            'actual_ratio_pct': np.asarray(actual_ratio_values, dtype=float),
        }

    return seed_ratios, seed_percentages, results


# ==========================================
# 5. 绘图：沿用 SIR 主实验样式
# ==========================================
def plot_jaccard_results(
    net,
    seed_percentages,
    results,
    methods,
    colors,
    markers,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x_vals = np.asarray(seed_percentages, dtype=float)

    for m in methods:
        if m not in results:
            continue

        y_vals = np.asarray(results[m]['Js'], dtype=float)

        lw, ms = (1.6, 4.5) if m == 'HOSH' else (1.2, 3.8)
        z_order, alpha_val = (10, 0.95) if m == 'HOSH' else (5, 0.88)

        ax.plot(
            x_vals,
            y_vals,
            linestyle='--',
            marker=markers.get(m, 'o'),
            color=colors.get(m, '#000000'),
            linewidth=lw,
            markersize=ms,
            markerfacecolor=colors.get(m, '#000000'),
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=alpha_val,
            zorder=z_order,
        )

    ax.set_title(net.capitalize(), fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('$p$ (%)', fontsize=11)
    ax.set_ylabel('$J_s$', fontsize=11)

    # 不固定为 1-10，因为小网络可能自动平移至 2-11 或 3-12。
    ax.set_xlim(x_vals[0] - 0.2, x_vals[-1] + 0.2)
    ax.set_xticks(x_vals.astype(int))

    # Jaccard 理论范围 [0,1]，但不强制 y 轴 0-1，
    # 以避免真实数值较小时曲线全部压缩在底部。
    finite_y = []
    for m in methods:
        if m in results:
            arr = np.asarray(results[m]['Js'], dtype=float)
            finite_y.extend(arr[np.isfinite(arr)].tolist())

    if finite_y:
        y_min = min(finite_y)
        y_max = max(finite_y)
        span = y_max - y_min
        pad = max(span * 0.08, 0.002)
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    plt.tight_layout(pad=0.2)

    pdf_path = os.path.join(output_dir, f"Jaccard_{net}.pdf")
    png_path = os.path.join(output_dir, f"Jaccard_{net}.png")

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png')
    plt.close()


# ==========================================
# 6. Excel 数据导出
# ==========================================
def build_network_long_table(
    network_name,
    n_nodes,
    seed_ratios,
    seed_percentages,
    results,
    methods,
):
    rows = []

    for idx, (ratio, nominal_pct) in enumerate(
        zip(seed_ratios, seed_percentages)
    ):
        for method_name in methods:
            rows.append({
                'Network': network_name,
                'N': n_nodes,
                'Nominal_Seed_Ratio_%': int(nominal_pct),
                'Nominal_Seed_Ratio': float(ratio),
                'Selected_k': int(results[method_name]['k'][idx]),
                'Actual_Selected_Ratio_%': float(
                    results[method_name]['actual_ratio_pct'][idx]
                ),
                'Method_Internal': method_name,
                'Method': DISPLAY_NAMES.get(method_name, method_name),
                'Average_Structural_Similarity_Js': float(results[method_name]['Js'][idx]),
            })

    return pd.DataFrame(rows)


def build_network_wide_table(
    seed_percentages,
    results,
    methods,
):
    data = {
        'Seed_Ratio_%': np.asarray(seed_percentages, dtype=int),
    }

    # k 对所有方法一致，因此只保存一列。
    first_method = methods[0]
    data['Selected_k'] = results[first_method]['k']
    data['Actual_Selected_Ratio_%'] = results[first_method]['actual_ratio_pct']

    for method_name in methods:
        display_name = DISPLAY_NAMES.get(method_name, method_name)
        data[f'{display_name}_Js'] = results[method_name]['Js']

    return pd.DataFrame(data)


# ==========================================
# 7. 主流程
# ==========================================
def main():
    set_seed(MASTER_SEED)

    output_dir = "results/exp_jaccard_structural_similarity"
    os.makedirs(output_dir, exist_ok=True)

    networks = get_network_list()
    methods = METHODS

    export_standalone_legend(
        methods,
        COLORS,
        MARKERS,
        output_dir,
    )

    all_long_dfs = []
    all_wide_dfs = []

    print("=" * 76)
    print("WCycle-style neighborhood Jaccard structural similarity experiment")
    print("Lower J_s indicates lower neighborhood overlap / structural redundancy.")
    print("Seed-ratio range is automatically shifted until k >= 2.")
    print("=" * 76)

    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")

        try:
            graph = download_and_load_graph(net)

            if graph is None or graph.number_of_nodes() == 0:
                print(f"    [Skip] {net}: graph failed to load or is empty.")
                continue

            n = graph.number_of_nodes()

            seed_ratios, seed_percentages, results = (
                exp_jaccard_structural_similarity(
                    methods,
                    graph,
                    net,
                )
            )

            # ---- 立即输出单网络图 ----
            plot_jaccard_results(
                net,
                seed_percentages,
                results,
                methods,
                COLORS,
                MARKERS,
                output_dir,
            )
            print(f"    [Output] PDF/PNG graph saved for {net}.")

            # ---- 单网络数据 ----
            df_long = build_network_long_table(
                net,
                n,
                seed_ratios,
                seed_percentages,
                results,
                methods,
            )

            df_wide = build_network_wide_table(
                seed_percentages,
                results,
                methods,
            )
            df_wide.insert(0, 'Network', net)

            excel_path = os.path.join(
                output_dir,
                f"Jaccard_AllData_{net}.xlsx",
            )

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_wide.to_excel(
                    writer,
                    sheet_name='Jaccard_Wide',
                    index=False,
                )
                df_long.to_excel(
                    writer,
                    sheet_name='Jaccard_Long',
                    index=False,
                )

            print(f"    [Output] Excel data saved: {excel_path}")

            all_long_dfs.append(df_long)
            all_wide_dfs.append(df_wide)

            # 每完成一个网络就更新一次总数据，便于中途查看。
            all_long_so_far = pd.concat(
                all_long_dfs,
                ignore_index=True,
            )
            all_long_so_far.to_excel(
                os.path.join(
                    output_dir,
                    "Jaccard_AllNetworks_Long.xlsx",
                ),
                index=False,
            )

            all_wide_so_far = pd.concat(
                all_wide_dfs,
                ignore_index=True,
            )
            all_wide_so_far.to_excel(
                os.path.join(
                    output_dir,
                    "Jaccard_AllNetworks_Wide.xlsx",
                ),
                index=False,
            )

            print("    [Output] Incremental all-network data updated.")

        except Exception as e:
            print(f"    [Error] Failed to process {net}: {e}")

    print("\n" + "=" * 76)
    print("Done.")
    print(f"Results directory: {output_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
