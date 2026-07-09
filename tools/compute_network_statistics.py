"""
计算网络的基本统计特性
包括：节点数、边数、平均度、平均最短路径长度、平均聚类系数、传播阈值
"""
import networkx as nx
import numpy as np
import pandas as pd
from network_loader import get_network_list, download_and_load_graph


def compute_epidemic_threshold(G):

    degrees = [d for n, d in G.degree()]
    k_avg = np.mean(degrees)
    k2_avg = np.mean([d ** 2 for d in degrees])

    if k2_avg == 0:
        return np.inf

    threshold = k_avg / (k2_avg-k_avg)
    return threshold


def compute_network_statistics(network_name):
    """
    计算单个网络的统计特性

    参数:
        network_name: 网络名称

    返回:
        包含统计信息的字典
    """
    print(f"\n处理网络: {network_name}")

    # 加载网络
    G = download_and_load_graph(network_name, verbose=False)

    if G is None:
        print(f"  ✗ 加载失败")
        return None

    # 计算基本统计量
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k_avg = 2 * m / n  # 平均度

    # 计算平均最短路径长度
    if nx.is_connected(G):
        d_avg = nx.average_shortest_path_length(G)
    else:
        # 如果网络不连通，计算最大连通分量的平均最短路径
        components = list(nx.connected_components(G))
        lcc = max(components, key=len)
        G_lcc = G.subgraph(lcc)
        d_avg = nx.average_shortest_path_length(G_lcc)

    # --- 新增：计算平均聚类系数 ---
    c_avg = nx.average_clustering(G)

    # 计算传播阈值
    beta_th = compute_epidemic_threshold(G)

    stats = {
        'network': network_name,
        'nodes': n,
        'edges': m,
        'k_avg': k_avg,
        'd_avg': d_avg,
        'c_avg': c_avg,  # 新增字段
        'beta_th': beta_th
    }

    print(f"  ✓ |V|={n}, |E|={m}, <k>={k_avg:.2f}, <d>={d_avg:.2f}, <C>={c_avg:.4f}, β_th={beta_th:.4f}")

    return stats


def main():
    """
    主函数：计算所有网络的统计特性并保存
    """
    print("=" * 70)
    print("网络统计特性计算")
    print("=" * 70)

    # 获取网络列表
    networks = get_network_list()
    print(f"\n共有 {len(networks)} 个网络需要处理")

    # 计算每个网络的统计信息
    results = []
    for network in networks:
        stats = compute_network_statistics(network)
        if stats:
            results.append(stats)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 按节点数排序
    df = df.sort_values('nodes')

    # 显示结果表格
    print("\n" + "=" * 70)
    print("统计结果汇总:")
    print("=" * 70)
    # 设置显示格式
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df.to_string(index=False))

    # 保存到Excel文件
    output_file = "network_statistics.xlsx"
    df.to_excel(output_file, index=False, float_format='%.4f')
    print(f"\n✓ 结果已保存到: {output_file}")

    # 显示LaTeX表格格式
    print("\n" + "=" * 70)
    print("LaTeX表格格式 (含聚类系数):")
    print("=" * 70)
    print("\\begin{table}[h]")
    print("\\centering")
    # 增加了一列 ccc (居中)
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    # 增加了 <C> 列
    print(
        "网络 & |V| & |E| & $\\langle k \\rangle$ & $\\langle d \\rangle$ & $\\langle C \\rangle$ & $\\beta_{th}$ \\\\")
    print("\\hline")

    for _, row in df.iterrows():
        # 增加了 row['c_avg'] 的输出
        print(f"{row['network']:15s} & {row['nodes']:5d} & {row['edges']:6d} & "
              f"{row['k_avg']:5.2f} & {row['d_avg']:5.2f} & {row['c_avg']:5.4f} & {row['beta_th']:6.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{九个真实网络的拓扑性质}")
    print("\\label{tab:network_stats}")
    print("\\end{table}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()