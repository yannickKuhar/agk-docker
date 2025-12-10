# import leidenalg
# import igraph as ig
# import networkx as nx

# from cdlib import algorithms


# class ModularityPartitioning:
#     def __init__(self, random_seed=42):
#         self.random_seed = random_seed
#
#     def louvain(self, G):
#         return nx.community.louvain_communities(G)
#
#     def greedy_modularity(self, G):
#         return nx.community.greedy_modularity_communities(G)
#
#     def naive_greedy_modularity(self, G):
#         return nx.community.naive_greedy_modularity_communities(G)
#
#     def laiden(self, G):
#         G_ig = ig.Graph.from_networkx(G.copy())
#         return leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
#
#     def eigenvector(self, G):
#         coms = algorithms.eigenvector(G)
#         return coms.communities
#
#     def paris(self, G):
#         coms = algorithms.paris(G)
#         return coms.communities
