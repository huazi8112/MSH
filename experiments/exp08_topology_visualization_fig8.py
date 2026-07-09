"""
实验: 网络拓扑可视化实验（独立子图）
模仿参考拓扑图的节点高亮方式，但每种方法使用独立子图显示。
绘图细节与 SIR 实验保持一致。
"""
import math
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from hosh_methods import get_node_scores
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings

# ==========================================
# 0. 基础配置（与 SIR 实验一致）
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


# ==========================================
# 1. 提取 Top-K 节点
# ==========================================
def extract_top_k_nodes(methods, g, top_k=50, network_name=None):
    """提取各方法 top-k 关键节点。"""
    print("  [Exp: Topology Visualization] Extracting Top-K nodes...")

    precomputed = load_precomputed_rankings(network_name) if network_name else None
    top_k_nodes = {}

    for method_name in methods:
        if precomputed and method_name in precomputed and precomputed[method_name]:
            scores = precomputed[method_name]
            print(f"    {method_name}: using precomputed rankings")
        else:
            scores = get_node_scores(method_name, g)
            print(f"    {method_name}: computing rankings on-the-fly")

        ranked_nodes = sorted(scores, key=scores.get, reverse=True)
        actual_k = min(top_k, g.number_of_nodes())
        top_k_nodes[method_name] = ranked_nodes[:actual_k]

    return top_k_nodes


# ==========================================
# 2. 绘图函数（独立子图）
# ==========================================
def plot_topology_subplots(net, g, top_k_nodes, methods, colors, markers, output_dir):
    """绘制多个基线方法独立小图 + 右侧 MSH 大图的网络拓扑可视化。"""
    print(f"    [Plotting] Generating independent subplots for {net}...")

    if g.number_of_nodes() == 0:
        return

    # 非连通图时仅展示最大连通子图，避免布局碎片化
    if not nx.is_connected(g):
        lcc = max(nx.connected_components(g), key=len)
        g_vis = g.subgraph(lcc).copy()
    else:
        g_vis = g.copy()

    iterations = 30 if g_vis.number_of_nodes() > 2000 else 50
    k_val = 1.0 / np.sqrt(max(g_vis.number_of_nodes(), 1))
    pos = nx.spring_layout(g_vis, seed=42, iterations=iterations, k=k_val)

    display_methods = [m for m in methods if m in top_k_nodes]
    if not display_methods:
        print("    [Skip] No methods available for visualization")
        return

    main_method = 'HOSH' if 'HOSH' in display_methods else display_methods[0]
    small_methods = [m for m in display_methods if m != main_method]

    # 动态非对称布局：左侧基线方法按 4 列排列。
    # 当最后一行不足 4 个方法时，将最后一行居中，避免出现明显空缺格。
    small_cols = 4
    small_rows = int(math.ceil(len(small_methods) / small_cols)) if small_methods else 1
    fig_width = 11.6
    fig_height = max(5.0, 1.65 * small_rows + 0.7)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 手动布局比 GridSpec 更适合 11 个基线方法：4 + 4 + 3，最后 3 个居中。
    left_x0 = 0.015
    left_width = 0.615
    right_x0 = 0.665
    right_width = 0.315
    bottom = 0.16
    top = 0.985
    hgap = 0.006
    vgap = 0.006
    panel_w = (left_width - hgap * (small_cols - 1)) / small_cols
    panel_h = (top - bottom - vgap * (small_rows - 1)) / small_rows

    axes_map = {}
    for idx, method in enumerate(small_methods):
        row = idx // small_cols
        col = idx % small_cols
        remaining = len(small_methods) - row * small_cols
        n_in_row = min(small_cols, remaining)
        # 若最后一行不足 small_cols 个，则水平居中。
        offset = (small_cols - n_in_row) / 2.0 if n_in_row < small_cols else 0.0
        x = left_x0 + (col + offset) * (panel_w + hgap)
        y = top - (row + 1) * panel_h - row * vgap
        axes_map[method] = fig.add_axes([x, y, panel_w, panel_h])

    # 主方法位于右侧大图，跨越全部高度。
    axes_map[main_method] = fig.add_axes([right_x0, bottom, right_width, top - bottom])

    bg_node_color = '#B5C4D6'
    bg_edge_color = '#E2E8F0'
    bg_alpha_lines = 0.4
    bg_alpha_nodes = 0.85

    for method, ax in axes_map.items():
        nx.draw_networkx_edges(
            g_vis,
            pos,
            ax=ax,
            edge_color=bg_edge_color,
            width=0.2,
            alpha=bg_alpha_lines,
        )
        nx.draw_networkx_nodes(
            g_vis,
            pos,
            ax=ax,
            node_color=bg_node_color,
            node_size=12.0,
            alpha=bg_alpha_nodes,
            linewidths=0,
        )

        seeds = [n for n in top_k_nodes[method] if n in g_vis]
        if seeds:
            node_size = 200.0 if method == main_method else 100.0
            edge_width = 1.0 if method == main_method else 0.8
            alpha_val = 0.98 if method == main_method else 0.95
            nx.draw_networkx_nodes(
                g_vis,
                pos,
                ax=ax,
                nodelist=seeds,
                node_color=colors.get(method, '#000000'),
                node_shape=markers.get(method, 'o'),
                node_size=node_size,
                edgecolors='black',
                linewidths=edge_width,
                alpha=alpha_val,
            )

        ax.axis('off')
        ax.set_aspect('equal')
        ax.margins(0.01)

    legend_elements = []
    legend_order = small_methods + [main_method]
    for m in legend_order:
        display_label = 'MSH' if m == 'HOSH' else m
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers.get(m, 'o'),
                color='w',
                markerfacecolor=colors.get(m, '#000000'),
                markeredgecolor='black',
                markeredgewidth=0.8,
                markersize=8.0,
                label=display_label,
            )
        )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.10),
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='black',
        framealpha=0.90,
        ncol=min(len(legend_order), 12),
        columnspacing=0.8,
        labelspacing=0.25,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.35,
        borderaxespad=0.0,
        fontsize=8.8,
    )

    # Manual add_axes layout is used above; no subplots_adjust is needed.

    save_pdf = os.path.join(output_dir, f"Topology_Subplots_{net}.pdf")
    save_png = os.path.join(output_dir, f"Topology_Subplots_{net}.png")

    fig.savefig(
        save_pdf,
        format='pdf',
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    fig.savefig(
        save_png,
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pil_kwargs={'quality': 95},
    )
    plt.close(fig)

    print(f"    [Output] High-quality figures saved: {save_pdf} and {save_png} (600 dpi)")

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_topology_results_to_excel(all_results, methods, top_k, output_dir):
    """保存各网络 top-k 节点到 Excel 便于复查。"""
    print("\n  [Export] Saving topology node lists to Excel...")

    for net, nodes_dict in all_results.items():
        df_data = {'Rank': np.arange(1, top_k + 1)}
        for m in methods:
            if m in nodes_dict:
                nodes = nodes_dict[m]
                padded_nodes = nodes + [None] * max(0, top_k - len(nodes))
                df_data[m] = padded_nodes[:top_k]

        df = pd.DataFrame(df_data)
        excel_path = os.path.join(output_dir, f"Topology_Nodes_{net}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Top_Nodes', index=False)
            worksheet = writer.sheets['Top_Nodes']
            worksheet.column_dimensions['A'].width = 10
            for i, _ in enumerate(methods):
                col_letter = chr(ord('B') + i)
                worksheet.column_dimensions[col_letter].width = 12

        print(f"    Saved: {excel_path}")


# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Network Topology Visualization (Independent Subplots)")
    print("=" * 60)

    set_seed(42)

    output_dir = "results/exp_topology_visualization"
    os.makedirs(output_dir, exist_ok=True)

    networks = get_network_list()
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

    all_results = {}
    top_k_count = 50

    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")
        print("-" * 60)

        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print("  [Skip] Network is empty or failed to load")
                continue

            n_nodes = g.number_of_nodes()
            print(f"  Nodes: {n_nodes}, Edges: {g.number_of_edges()}")

            if n_nodes > 5000:
                print("  [Warning] Network >5000 nodes, skipping for visualization clarity and speed")
                continue

            top_k_nodes = extract_top_k_nodes(methods, g, top_k=top_k_count, network_name=net)
            plot_topology_subplots(net, g, top_k_nodes, methods, colors, markers, output_dir)

            all_results[net] = top_k_nodes

        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        save_topology_results_to_excel(all_results, methods, top_k_count, output_dir)

    print("\n" + "=" * 60)
    print(" Experiment: Network Topology Visualization Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
