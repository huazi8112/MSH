import networkx as nx

from msh_methods import calculate_voterank


def rank_order(scores):
    return [node for node, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def test_voterank_equal_initial_scores_choose_smallest_node_id():
    # In a 4-cycle all nodes have the same initial voting score. The explicit
    # internal tie rule must select node 0 first after integer relabeling.
    g = nx.cycle_graph(4)
    order = rank_order(calculate_voterank(g))
    assert order[0] == 0


def test_voterank_is_deterministic():
    g = nx.path_graph(8)
    assert calculate_voterank(g) == calculate_voterank(g)
