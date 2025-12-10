import networkx as nx


class TreeDecompositionPartitioning:
    def __init__(self):
        pass

    def treewidth_min_degree(self, G):
        tw, T = nx.algorithms.approximation.treewidth_min_degree(G)
        return T

    def treewidth_min_fill_in(self, G):
        tw, T = nx.algorithms.approximation.treewidth_min_fill_in(G)
        return T
