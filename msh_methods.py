"""Core scoring methods for the revised MSH repository.

MSH denotes Mesoscopic Structural Holes, a clique-aware structural-hole
measure for simple, unweighted, undirected pairwise graphs.  The file also
implements the baseline methods and ablation variants used in the revised
manuscript.

Note on identifiers: during revision, some scripts and caches used the legacy
internal prefix ``HOSH``.  Public figures/tables use ``MSH``.  The dispatcher
therefore accepts both prefixes and maps MSH aliases to the legacy internal
variant names when necessary.
"""
import math
from collections import defaultdict

import networkx as nx
import numpy as np
try:
    import igraph as ig
except Exception:  # pragma: no cover - fallback backend
    ig = None

try:
    import community.community_louvain as community_louvain
except Exception:  # pragma: no cover - CHBC needs python-louvain
    community_louvain = None


# ==========================================
# 0. Maximal-clique helper functions
# ==========================================
def enumerate_maximal_cliques(g):
    """
    Enumerate maximal cliques while preserving original NetworkX node labels.

    igraph is used as the fast backend. We construct the igraph object manually
    instead of relying on Graph.from_networkx(), because the latter may return
    igraph vertex indices that are unsafe when NetworkX node labels are not
    consecutive integers.
    """
    nodes = list(g.nodes())
    if not nodes:
        return []

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()]

    if ig is not None:
        try:
            g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
            clique_indices = g_ig.maximal_cliques()
            return [tuple(nodes[i] for i in c) for c in clique_indices]
        except Exception:
            pass

    # Fallback for environments without igraph or unusual graph objects.
    return [tuple(c) for c in nx.find_cliques(g)]


def build_clique_data(g):
    """Build reusable maximal-clique data for MSH and clique-based baselines."""
    cliques = enumerate_maximal_cliques(g)
    clique_sets = [set(c) for c in cliques]
    clique_sizes = [len(c) for c in cliques]

    node_cliques_map = {n: [] for n in g.nodes()}
    for idx, c_nodes in enumerate(cliques):
        for node in c_nodes:
            if node in node_cliques_map:
                node_cliques_map[node].append(idx)

    return {
        'cliques': cliques,
        'node_cliques_map': node_cliques_map,
        'clique_sizes': clique_sizes,
        'clique_sets': clique_sets,
    }


def _normalize_clique_data(g, clique_data=None):
    """
    Accept externally supplied clique_data but make sure all required fields exist.
    This keeps compatibility with older scripts that pass a partial clique cache.
    """
    if clique_data is None:
        return build_clique_data(g)

    cliques = [tuple(c) for c in clique_data.get('cliques', [])]
    clique_sizes = list(clique_data.get('clique_sizes', []))
    if not clique_sizes:
        clique_sizes = [len(c) for c in cliques]

    clique_sets = list(clique_data.get('clique_sets', []))
    if not clique_sets:
        clique_sets = [set(c) for c in cliques]
    else:
        clique_sets = [set(c) for c in clique_sets]

    raw_map = clique_data.get('node_cliques_map', {})
    node_cliques_map = {n: list(raw_map.get(n, [])) for n in g.nodes()}

    # If the map is missing or incomplete, rebuild it from cliques.
    if not raw_map or sum(len(v) for v in node_cliques_map.values()) == 0:
        node_cliques_map = {n: [] for n in g.nodes()}
        for idx, c_nodes in enumerate(cliques):
            for node in c_nodes:
                if node in node_cliques_map:
                    node_cliques_map[node].append(idx)

    return {
        'cliques': cliques,
        'node_cliques_map': node_cliques_map,
        'clique_sizes': clique_sizes,
        'clique_sets': clique_sets,
    }


# ==========================================
# 1. MSH and ablation variants
# ==========================================
def _external_transform(x, variant):
    """Scalar external-capability transformation used by MSH variants."""
    if x <= 0:
        return 0.0
    if variant == 'HOSH-Lin':
        return float(x)
    if variant == 'HOSH-BoxCox':
        return 2.0 * (math.sqrt(1.0 + x) - 1.0)
    return math.log1p(x)


def _build_directed_overlap_cache(node_cliques_map, clique_sizes):
    """
    Compute directed clique-overlap coefficients without repeated set intersections.

    For two cliques beta and alpha:
        o_{beta,alpha} = max(0, |C_beta ∩ C_alpha| - 1) / (|C_beta| - 1)

    The intersection size is obtained by counting how many nodes belong to each
    unordered pair of maximal cliques. This is exact and avoids the expensive
    repeated clique_sets[beta].intersection(clique_sets[alpha]) calls inside the
    node-level scoring loop.
    """
    pair_shared_count = defaultdict(int)

    for clique_indices in node_cliques_map.values():
        L = len(clique_indices)
        if L <= 1:
            continue
        # Clique lists are already small for most sparse networks; explicit loops
        # avoid constructing many temporary tuples from itertools.combinations.
        for i in range(L - 1):
            a = clique_indices[i]
            # A maximal clique of size 2 cannot have a non-zero redundancy
            # overlap with any other distinct maximal clique after the -1
            # correction; otherwise it would be contained in a larger clique.
            if clique_sizes[a] <= 2:
                continue
            for j in range(i + 1, L):
                b = clique_indices[j]
                if clique_sizes[b] <= 2:
                    continue
                if a < b:
                    pair_shared_count[(a, b)] += 1
                else:
                    pair_shared_count[(b, a)] += 1

    directed_overlap = {}
    for (a, b), shared_size in pair_shared_count.items():
        numerator = shared_size - 1
        if numerator <= 0:
            continue

        denom_a = clique_sizes[a] - 1
        denom_b = clique_sizes[b] - 1
        if denom_a > 0:
            directed_overlap[(a, b)] = numerator / denom_a  # beta=a, alpha=b
        if denom_b > 0:
            directed_overlap[(b, a)] = numerator / denom_b  # beta=b, alpha=a

    return directed_overlap


def calculate_hosh_unified(g, variant='HOSH', clique_data=None, xi=0.001):
    """
    Optimized MSH and ablation variants.

    The mathematical output is kept consistent with the original implementation,
    but the overlap-redundancy stage is accelerated by precomputing clique-pair
    overlaps from the node-clique incidence structure.
    """
    if variant == 'HOSH-Sqrt':
        variant = 'HOSH-BoxCox'

    data = _normalize_clique_data(g, clique_data)
    cliques = data['cliques']
    node_cliques_map = data['node_cliques_map']
    clique_sizes = data['clique_sizes']

    nodes = list(g.nodes())
    degrees = dict(g.degree())
    scores = {n: 0.0 for n in nodes}

    if not cliques:
        return scores

    # --- MSH-E: external-degree-only ablation ---
    if variant == 'HOSH-E':
        for v in nodes:
            deg_v = degrees[v]
            total = 0.0
            for alpha in node_cliques_map.get(v, []):
                total += max(0.0, deg_v - (clique_sizes[alpha] - 1))
            scores[v] = total
        return scores

    needs_external_adjustment = variant not in ('HOSH-C', 'HOSH-NE')
    needs_overlap_redundancy = variant not in ('HOSH-C', 'HOSH-NO')

    # --- Step 1: external capability per node-clique incidence ---
    clique_node_k_totals = None
    clique_norm_denominators = None

    if needs_external_adjustment:
        clique_node_k_totals = [None] * len(cliques)
        clique_norm_denominators = [xi] * len(cliques)

        for alpha, c_nodes in enumerate(cliques):
            c_size = clique_sizes[alpha]
            base = c_size - 1

            ext_values = []
            total_ext = 0.0
            for node in c_nodes:
                val = max(0.0, degrees[node] - base)
                ext_values.append((node, val))
                total_ext += val

            k_map = {}
            max_k = 0.0
            sum_k = 0.0
            if c_size > 1:
                for node, node_ext in ext_values:
                    implicit_capability = _external_transform(total_ext - node_ext, variant)
                    k_val = node_ext + implicit_capability
                    k_map[node] = k_val
                    sum_k += k_val
                    if k_val > max_k:
                        max_k = k_val
            else:
                for node, node_ext in ext_values:
                    k_map[node] = node_ext
                    sum_k += node_ext
                    if node_ext > max_k:
                        max_k = node_ext

            clique_node_k_totals[alpha] = k_map
            if variant == 'HOSH-SumNorm':
                clique_norm_denominators[alpha] = sum_k + xi
            else:
                clique_norm_denominators[alpha] = max_k + xi

    # --- Step 2: effective dependence p* ---
    node_p_stars = {}
    for v in nodes:
        my_indices = node_cliques_map.get(v, [])
        if not my_indices:
            continue

        p_base = 1.0 / len(my_indices)
        p_dict = {}

        if not needs_external_adjustment:
            for alpha in my_indices:
                p_dict[alpha] = p_base
        else:
            for alpha in my_indices:
                k_total_i = clique_node_k_totals[alpha][v]
                norm_denominator = clique_norm_denominators[alpha]
                autonomy_factor = k_total_i / norm_denominator
                p_dict[alpha] = p_base * (1.0 - autonomy_factor)

        node_p_stars[v] = p_dict

    # --- Step 3: overlap redundancy and final constraint ---
    overlap_by_beta = None
    if needs_overlap_redundancy:
        directed_overlap = _build_directed_overlap_cache(node_cliques_map, clique_sizes)
        overlap_by_beta = defaultdict(list)
        for (beta, alpha), weight in directed_overlap.items():
            overlap_by_beta[beta].append((alpha, weight))

    for v in nodes:
        my_indices = node_cliques_map.get(v, [])
        if not my_indices:
            continue

        p_dict = node_p_stars[v]
        total_constraint = 0.0

        if not needs_overlap_redundancy:
            for p_val in p_dict.values():
                total_constraint += p_val * p_val
        else:
            # Accumulate only non-zero overlap terms. This avoids an O(L_v^2)
            # loop over clique pairs that share only the focal node and therefore
            # have zero redundancy after the "-1" correction.
            my_set = set(my_indices)
            indirect_by_alpha = {alpha: 0.0 for alpha in my_indices}

            for beta, p_beta in p_dict.items():
                for alpha, weight in overlap_by_beta.get(beta, ()):
                    if alpha in my_set:
                        indirect_by_alpha[alpha] += p_beta * weight

            for alpha in my_indices:
                val = p_dict[alpha] + indirect_by_alpha[alpha]
                total_constraint += val * val

        scores[v] = 1.0 - total_constraint

    return scores


# ==========================================
# 2. 其他 Baseline 方法保持不变
# ==========================================

def calculate_ish(g):
    scores = {}
    degrees = dict(g.degree())
    for i in g.nodes():
        k_i = degrees[i]
        if k_i == 0:
            scores[i] = 0.0
            continue
        edge_weights = {j: k_i + degrees[j] for j in g.neighbors(i)}
        w_i = sum(edge_weights.values())
        if w_i == 0:
            scores[i] = 0.0
            continue
        relative_importance = {j: edge_weights[j] / w_i for j in g.neighbors(i)}
        sum_p_ij = sum(relative_importance.values())
        if sum_p_ij > 0:
            for j in relative_importance: relative_importance[j] /= sum_p_ij
        constraint = 0.0
        for j in g.neighbors(i):
            p_ij = relative_importance[j]
            indirect = 0.0
            for q in g.neighbors(i):
                if q == j: continue
                p_iq = relative_importance[q]
                if g.has_edge(q, j):
                    k_q = degrees[q]
                    w_q = sum(k_q + degrees[neighbor] for neighbor in g.neighbors(q))
                    if w_q > 0:
                        p_qj = (k_q + degrees[j]) / w_q
                        sum_p_qn = sum((k_q + degrees[n]) / w_q for n in g.neighbors(q))
                        if sum_p_qn > 0: p_qj /= sum_p_qn
                        indirect += p_iq * p_qj
            constraint += (p_ij + indirect) ** 2
        scores[i] = 1.0 - constraint
    return scores


def calculate_sh(g):
    scores = {}
    degrees = dict(g.degree())
    for i in g.nodes():
        k_i = degrees[i]
        if k_i == 0:
            scores[i] = 0.0
            continue
        neighbors_i = set(g.neighbors(i))
        p_ij_base = 1.0 / k_i
        constraint = 0.0
        for j in neighbors_i:
            p_ij = p_ij_base
            indirect = 0.0
            for q in neighbors_i:
                if q == j: continue
                if g.has_edge(q, j):
                    k_q = degrees[q]
                    if k_q > 0: indirect += p_ij_base * (1.0 / k_q)
            constraint += (p_ij + indirect) ** 2
        scores[i] = 1.0 - constraint
    return scores


def calculate_ci(g, radius=2):
    scores = {}
    degrees = dict(g.degree())
    for i in g.nodes():
        if degrees[i] <= 1:
            scores[i] = 0.0
            continue
        visited = {i}
        current_level = [i]
        for _ in range(radius):
            next_level = []
            for node in current_level:
                for neighbor in g.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level
        sum_kj_minus_1 = sum(max(0, degrees[j] - 1) for j in current_level)
        scores[i] = (degrees[i] - 1) * sum_kj_minus_1
    return scores


def calculate_snc(g):
    scores = {}
    degrees = dict(g.degree())
    clustering = nx.clustering(g)
    sh_scores = calculate_sh(g)
    mass = {}
    omega = {}
    for i in g.nodes():
        constraint_i = 1.0 - sh_scores.get(i, 0.0)
        mass[i] = degrees[i] * np.exp(clustering.get(i, 0.0))
        omega[i] = np.exp(-constraint_i) / 2.0
    for i in g.nodes():
        neighbors_1 = set(g.neighbors(i))
        neighbors_2 = set()
        for n1 in neighbors_1: neighbors_2.update(g.neighbors(n1))
        neighborhood = neighbors_1.union(neighbors_2)
        neighborhood.add(i)
        nc_i = sum(mass[j] * mass[i] * omega[j] for j in neighborhood)
        scores[i] = nc_i * omega[i]
    return scores


def calculate_voterank(g):
    """VoteRank baseline with deterministic internal tie-breaking.

    The voting update follows the standard VoteRank procedure used in the
    revision: initial voting ability is 1, the neighbor decrement is
    ``1 / <k>``, and the selected node's voting ability is set to 0.

    Reproducibility rule
    --------------------
    If multiple candidates have the same voting score at an iterative
    selection step, the node with the smallest node ID is selected. The
    empirical preprocessing pipeline relabels nodes to consecutive integers,
    so this implements the ascending-node-ID rule reported in Supplementary
    Table S3.
    """
    N = g.number_of_nodes()
    scores = {n: 0.0 for n in g.nodes()}
    if N == 0:
        return scores

    avg_degree = sum(dict(g.degree()).values()) / N if N > 0 else 0.0
    f = 1.0 / avg_degree if avg_degree > 0 else 0.0
    voting_ability = {n: 1.0 for n in g.nodes()}
    candidates = set(g.nodes())
    rank = 0
    neighbors_map = {n: list(g.neighbors(n)) for n in g.nodes()}

    while candidates:
        best_node = None
        max_score = -1.0
        for node in candidates:
            current_score = sum(voting_ability[neighbor] for neighbor in neighbors_map[node])
            if (
                current_score > max_score
                or (current_score == max_score and (best_node is None or node < best_node))
            ):
                max_score = current_score
                best_node = node

        scores[best_node] = float(N - rank)
        rank += 1
        candidates.remove(best_node)
        voting_ability[best_node] = 0.0
        for neighbor in neighbors_map[best_node]:
            if voting_ability[neighbor] > 0:
                voting_ability[neighbor] = max(0.0, voting_ability[neighbor] - f)

    return scores


def calculate_snim(g, alpha=3, clique_data=None):
    """SNIM baseline. Following the original setting, only maximal cliques with size >= alpha are used."""
    data = _normalize_clique_data(g, clique_data)
    filtered_cliques = [c for c in data['cliques'] if len(c) >= alpha]
    scores = {n: 0.0 for n in g.nodes()}
    if not filtered_cliques:
        return scores

    node_clique_counts = {n: 0 for n in g.nodes()}
    node_clique_unions = {n: set() for n in g.nodes()}
    for c_nodes in filtered_cliques:
        c_set = set(c_nodes)
        for node in c_nodes:
            if node in node_clique_counts:
                node_clique_counts[node] += 1
                node_clique_unions[node].update(c_set)

    for node in g.nodes():
        if node_clique_counts[node] > 0:
            scores[node] = float(node_clique_counts[node] * len(node_clique_unions[node]))
    return scores

def get_network_partition(g, seed=42):
    if community_louvain is None:
        raise ImportError('CHBC requires the python-louvain package: community.community_louvain')
    partition = community_louvain.best_partition(g, random_state=seed)
    comm_size_map = {}
    for comm_id in partition.values():
        comm_size_map[comm_id] = comm_size_map.get(comm_id, 0) + 1
    return partition, comm_size_map


def calculate_chbc(g, partition, comm_size_map):
    scores = {n: 0.0 for n in g.nodes()}
    if g.number_of_nodes() == 0: return scores
    for i in g.nodes():
        my_comm_id = partition[i]
        n_cq = comm_size_map[my_comm_id]
        d_intra = 0
        d_inter = 0
        adj_communities = set()
        for j in g.neighbors(i):
            neighbor_comm_id = partition[j]
            if neighbor_comm_id == my_comm_id:
                d_intra += 1
            else:
                d_inter += 1
                adj_communities.add(neighbor_comm_id)
        scores[i] = float((n_cq * d_intra) + (d_inter * len(adj_communities)))
    return scores


# ==========================================
# 3. 算法调用总路由
# ==========================================

METHOD_ALIASES = {
    'MSH': 'HOSH',
    'MSH-NO': 'HOSH-NO',
    'MSH-NE': 'HOSH-NE',
    'MSH-E': 'HOSH-E',
    'MSH-C': 'HOSH-C',
    'MSH-Lin': 'HOSH-Lin',
    'MSH-BoxCox': 'HOSH-BoxCox',
    'MSH-AltConcave': 'HOSH-BoxCox',
    'MSH-Sqrt': 'HOSH-Sqrt',
    'MSH-SumNorm': 'HOSH-SumNorm',
}

def _internal_method_name(method):
    return METHOD_ALIASES.get(method, method)

def get_node_scores(method, g, partition=None, comm_size_map=None, clique_data=None):
    """
    Unified method dispatcher. Accepts both public MSH identifiers and
    legacy internal HOSH identifiers used by earlier caches/scripts.
    """
    method = _internal_method_name(method)
    if method.startswith('HOSH'):
        return calculate_hosh_unified(g, variant=method, clique_data=clique_data)

    elif method == 'ISH':
        return calculate_ish(g)
    elif method == 'DC':
        return nx.degree_centrality(g)
    elif method == 'BC':
        return nx.betweenness_centrality(g)
    elif method == 'CC':
        return nx.closeness_centrality(g)
    elif method == 'K-Shell':
        return dict(nx.core_number(g))
    elif method == 'SH':
        return calculate_sh(g)
    elif method == 'CI':
        return calculate_ci(g)
    elif method == 'SNC':
        return calculate_snc(g)
    elif method == 'VoteRank':
        return calculate_voterank(g)
    elif method == 'SNIM':
        return calculate_snim(g, alpha=3, clique_data=clique_data)
    elif method == 'CHBC':
        if partition is None or comm_size_map is None:
            partition, comm_size_map = get_network_partition(g)
        return calculate_chbc(g, partition, comm_size_map)
    else:
        raise ValueError(f"Unknown Method: {method}")