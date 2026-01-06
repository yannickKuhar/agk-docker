import time
import random
import numpy as np
import pandas as pd
import networkx as nx

from rdkit import Chem
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


class Utility:
    @staticmethod
    def parse_dataset_to_nx(graph_labels_file, graph_edges_file, graph_indicator_file):
        ngc = dict()

        with open(graph_labels_file, "r") as f:
            graph_labels = [int(x.strip()) for x in f.readlines()]

        with open(graph_edges_file, "r") as f:
            graph_edges = [x.strip().split(",") for x in f.readlines()]

        with open(graph_indicator_file, "r") as f:
            for node, graph in enumerate(f.readlines()):
                ngc[node + 1] = int(graph)

        graphs = {i + 1: nx.Graph() for i in range((len(graph_labels)))}

        for edge in graph_edges:
            u = int(edge[0])
            v = int(edge[1])

            graphs[ngc[u]].add_edge(u, v)

        return list(graphs.values()), graph_labels

    @staticmethod
    def parse_dataset(graph_labels_file, graph_edges_file, graph_indicator_file):
        # node_labels_file = f"./data/{dataset}/{dataset}_node_labels.txt"
        # edge_labels_file = f"./data/{dataset}/{dataset}_edge_labels.txt"
        # graph_labels_file = f"../data/{dataset}/{dataset}_graph_labels.txt"
        # graph_edges_file = f"../data/{dataset}/{dataset}_A.txt"
        # graph_indicator_file = f"../data/{dataset}/{dataset}_graph_indicator.txt"

        # node graph correspondence
        ngc = dict()
        # edge line correspondence
        elc = dict()

        with open(graph_labels_file, "r") as f:
            graph_labels = [int(x.strip()) for x in f.readlines()]

        with open(graph_edges_file, "r") as f:
            graph_edges = []
            for id, x in enumerate(f.readlines()):
                edge = x.strip().split(",")
                graph_edges.append(edge)
                elc[id + 1] = edge

        with open(graph_indicator_file, "r") as f:
            for node, graph in enumerate(f.readlines()):
                ngc[node + 1] = int(graph)

        graphs = {i + 1: [set(), dict(), dict()] for i in range((len(graph_labels)))}

        # with open(node_labels_file, "r") as f:
        #     for node, label in enumerate(f.readlines()):
        #         graphs[ngc[node + 1]][1][node + 1] = int(label)

        for edge in graph_edges:
            u = int(edge[0])
            v = int(edge[1])

            graphs[ngc[u]][0].add((u, v))
            graphs[ngc[u]][0].add((v, u))

        return list(graphs.values()), graph_labels

    @staticmethod
    def smiles_to_nx(smiles: str) -> nx.Graph:
        # Parse with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        G = nx.Graph()

        # Add atoms as nodes
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            G.add_node(idx,
                       symbol=atom.GetSymbol(),
                       atomic_num=atom.GetAtomicNum(),
                       formal_charge=atom.GetFormalCharge())

        # Add bonds as edges
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            G.add_edge(a1, a2,
                       bond_type=str(bond.GetBondType()),
                       is_aromatic=bond.GetIsAromatic())

        return G

    @staticmethod
    def get_graphlet_index():
        return list(range(29))

    @staticmethod
    def get_symmetry_index():
        return list(range(13))

    @staticmethod
    def get_random_graplet_and_vertex_symmetry_subset():
        def coin_flip_subset(full_set):
            result = []

            for i in full_set:
                if random.randint(0, 1) == 1:
                    result.append(i)

            return result

        graphlet_index = list(range(29))
        symmetry_index = list(range(13))

        graphlet_subset = coin_flip_subset(graphlet_index)
        symmetry_subset = coin_flip_subset(symmetry_index)

        return graphlet_subset, symmetry_subset

    @staticmethod
    def get_pd(x, y, dataset, k):
        data = []

        for vec, label in zip(x, y):
            data.append(np.append(vec, label))

        names = [f"{i}" for i in range(len(data[0]) - 1)]
        names.append("class")

        data = np.array(data)

        df = pd.DataFrame(data=data, columns=names)
        df.to_csv(f"data_csv/{dataset}_{type(k).__name__}.csv")

        return df

    @staticmethod
    def save_results(path, results, name):
        with open(f"{path}/results.csv", "w+") as f:
            f.write(f"Dataset, MC Score, {name} Time, {name} Time Uncertainty, {name} Accuracy,"
                    f"{name} Accuracy Uncertainty, {name} F1, {name} F1 Uncertainty,"
                    f"f{name} Fidelity, f{name} Fidelity Uncertainty\n")

            for r in results:
                f.write(",".join([str(x) for x in r]) + "\n")

    @staticmethod
    def boostrap_sample_graphs(graphs, labels):
        objects = list(zip(graphs, labels))
        samples = random.choices(objects, k=len(objects))

        gs = []
        ls = []

        for g, l in samples:
            gs.append(g)
            ls.append(l)

        return gs, ls

    @staticmethod
    def bootsrap_sample(X_train, y_train):
        n_train = len(X_train)
        indices = np.random.choice(n_train, n_train, replace=True)

        # print(X_train)

        # X_train = np.array(X_train)
        # y_train = np.array(y_train)

        X = X_train[indices]
        y = y_train[indices]

        return X, y

    @staticmethod
    def get_mc(X, y):
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X, y)

        return round(dummy_clf.score(X, y), 4)

    @staticmethod
    def grakel_to_networkx(graph_db):
        graphs = []

        for g in graph_db:
            G = nx.Graph()

            for node in g[1]:
                G.add_node(node, label=g[1][node])

            for edge in g[0]:
                G.add_edge(edge[0], edge[1])

            graphs.append(G)

        return graphs

    @staticmethod
    def read_csv(file_x, file_y):
        X = np.loadtxt(open(file_x, "rb"), delimiter=",")
        y = np.loadtxt(open(file_y, "rb"), delimiter=",")
        return X, y
