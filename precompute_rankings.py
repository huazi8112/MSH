"""
预计算节点排名脚本 (全面升级版 - 包含极大团缓存与 7 大消融变体)
1. 提前计算所有网络的所有方法的节点排名分数并保存到本地
2. 引入【极大团全局缓存机制】，避免 HOSH 及其 7 个消融变体重复计算极大团，计算提速 10 倍以上
3. 包含严格的 ID-based 平局打破接口
"""
import os
import pickle
import numpy as np
import igraph as ig
from tqdm import tqdm
from network_loader import download_and_load_graph, get_network_list
from hosh_methods import get_node_scores, get_network_partition

# ================= 目录配置 =================
OUTPUT_DIR = "results/node_rankings"
CLIQUE_CACHE_DIR = "results/clique_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIQUE_CACHE_DIR, exist_ok=True)

# ================= 方法定义 =================
# 包含 Baseline、结构组件消融、数学公式消融
METHODS = [
    # 1. 核心与基线算法 (Core & Baselines)
    'HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC', 'VoteRank', 'SNIM', 'CHBC',

    # 2. 阶段一：结构组件消融 (Structural Component Ablation)
    'HOSH-C',    # 仅团成员身份 (仅基础依赖)
    'HOSH-E',    # 仅外部度 (无视最大团结构)
    'HOSH-NE',   # 无外部度调整 (不使用节点在团内的相对自主性)
    'HOSH-NO',   # 无重叠冗余计算 (假设所有团之间相互独立)

    # 3. 阶段二：数学函数形态消融 (Mathematical Function Ablation)
    'HOSH-Lin',      # 线性惩罚 (替换对数惩罚，无衰减)
    'HOSH-Sqrt',     # 根号惩罚 (替换对数惩罚，中度衰减)
    'HOSH-SumNorm',  # 综合求和归一化 (替换最大值Max归一化)
]

# ================= 核心缓存逻辑 =================

def load_or_compute_clique_cache(network_name, g):
    """
    加载或计算极大团全局缓存。
    为 HOSH 系列算法提供统一的拓扑基础数据，避免 NP-Hard 问题重复计算。
    """
    cache_file = os.path.join(CLIQUE_CACHE_DIR, f"{network_name}_cliques.pkl")

    if os.path.exists(cache_file):
        print(f"  ✓ 极大团数据: 已从缓存加载 ({cache_file})")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"  > 极大团数据: 未命中缓存，开始使用 igraph 枚举 (可能会耗时)...")

    # 使用 igraph 进行高效的极大团枚举
    g_ig = ig.Graph.from_networkx(g)
    cliques = g_ig.maximal_cliques()

    # 预计算 HOSH 需要的数据结构 (空间换时间)
    node_cliques_map = {n: [] for n in g.nodes()}
    for idx, c in enumerate(cliques):
        for n in c:
            node_cliques_map[n].append(idx)

    clique_data = {
        'cliques': cliques,
        'node_cliques_map': node_cliques_map,
        'clique_sizes': [len(c) for c in cliques],
        'clique_sets': [set(c) for c in cliques]
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(clique_data, f)
    print(f"  ✓ 极大团数据: 计算完毕并已缓存 ({len(cliques)} 个极大团)")

    return clique_data

# ================= 计算与保存逻辑 =================

def compute_and_save_rankings(network_name, methods=None, force_recompute_method=False):
    if methods is None:
        methods = METHODS

    print(f"\n{'=' * 60}")
    print(f"正在处理网络: {network_name}")
    print(f"{'=' * 60}")

    try:
        g = download_and_load_graph(network_name, verbose=False)
        print(f"✓ 网络加载成功: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边")
    except Exception as e:
        print(f"✗ 网络加载失败: {e}")
        return None

    # 1. 提前加载或计算极大团缓存 (供 HOSH 系列使用)
    clique_data = load_or_compute_clique_cache(network_name, g)

    # 2. 提前加载或计算社区划分缓存 (供 CHBC 等 Baseline 使用)
    partition, comm_size_map = get_network_partition(g)

    # 3. 读取已有的排名缓存，增量更新
    rankings = load_precomputed_rankings(network_name)
    if rankings is None:
        rankings = {}

    updated = False
    for method in tqdm(methods, desc=f"计算 {network_name}", ncols=80):
        # 增量跳过逻辑
        if method in rankings and rankings[method] is not None and not force_recompute_method:
            print(f"  ⏭ {method:15s}: 已缓存，跳过")
            continue

        try:
            # 核心修改：将 clique_data 传入接口，底层直接读取，避免重复枚举团
            scores = get_node_scores(
                method, g,
                partition=partition,
                comm_size_map=comm_size_map,
                clique_data=clique_data  # 关键新增参数
            )
            rankings[method] = scores
            updated = True
            print(f"  ✓ {method:15s}: 计算完成 ({len(scores)} 个节点)")
        except Exception as e:
            print(f"  ✗ {method:15s}: 计算失败 - {e}")
            if method not in rankings:
                rankings[method] = None

    # 落盘保存
    if updated:
        output_file = os.path.join(OUTPUT_DIR, f"{network_name}_rankings.pkl")
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(rankings, f)
            print(f"✓ 结果已增量更新至: {output_file}")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return None
    else:
        print(f"✓ 所有指定方法均已存在，无需更新文件。")

    return rankings

def load_precomputed_rankings(network_name):
    """加载预计算的节点排名分数"""
    file_path = os.path.join(OUTPUT_DIR, f"{network_name}_rankings.pkl")
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"✗ 加载 {network_name} 的预计算结果失败: {e}")
        return None

def precompute_all_networks(networks=None, methods=None, force_recompute=False):
    if networks is None:
        networks = get_network_list()
    if methods is None:
        methods = METHODS

    print(f"\n{'#' * 70}")
    print(f"# 增量预计算节点排名任务")
    print(f"# 网络数量: {len(networks)}")
    print(f"# 包含变体: {len(methods)} 个")
    print(f"{'#' * 70}\n")

    results_summary = {}
    for i, network in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] 检查网络: {network}")
        rankings = compute_and_save_rankings(network, methods, force_recompute_method=force_recompute)
        results_summary[network] = "完毕" if rankings is not None else "失败"

    print(f"\n\n{'=' * 70}")
    print("增量预计算任务完成!")

def get_standardized_ranked_nodes(scores, round_decimals=8):
    """
    【标准化的平局处理接口】
    针对审稿人要求，采用 ID-based 绝对中立的确定性平局打破策略。
    """
    nodes = list(scores.keys())
    nodes.sort(key=lambda n: (-round(scores[n], round_decimals), n))
    return nodes

def verify_rankings_file(network_name):
    """验证预计算文件的完整性"""
    rankings = load_precomputed_rankings(network_name)
    if rankings is None:
        print(f"✗ {network_name}: 文件不存在")
        return False

    print(f"\n检查 {network_name}:")
    all_valid = True
    for method in METHODS:
        if method not in rankings:
            print(f"  ✗ {method}: 缺失")
            all_valid = False
        elif rankings[method] is None:
            print(f"  ✗ {method}: 数据为None")
            all_valid = False
        else:
            print(f"  ✓ {method:15s}: {len(rankings[method])} 个节点")
    return all_valid

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="预计算节点排名脚本")
    parser.add_argument('--networks', nargs='+', default=None, help='指定要计算的网络列表')
    parser.add_argument('--methods', nargs='+', default=None, help='指定要计算的方法列表')
    parser.add_argument('--force', action='store_true', help='强制重新计算')
    parser.add_argument('--verify', nargs='+', default=None, help='验证指定网络的预计算文件')

    args = parser.parse_args()

    if args.verify:
        print("\n验证预计算文件:")
        for network in args.verify:
            verify_rankings_file(network)
    else:
        precompute_all_networks(
            networks=args.networks,
            methods=args.methods,
            force_recompute=args.force
        )