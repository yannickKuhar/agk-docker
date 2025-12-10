import time
import numpy as np
import networkx as nx


class BaseKernel:
    def __init__(self):
        self.phi_x = None
        self.phi_y = None
        self.all_keys = None

    def grakel_to_networkx(self, graph_db):
        graphs = []

        for g in graph_db:
            G = nx.Graph()

            for edge in g[0]:
                G.add_edge(edge[0], edge[1])

            graphs.append(G)

        return graphs

    def min_max_normalize(self, X_train, X_test):
        min = np.min(X_train)
        max = np.max(X_train)

        X_train = (X_train - min) / ((max - min) + 0.000000001)
        X_test = (X_test - min) / ((max - min) + 0.000000001)

        return X_train, X_test

    def make_vectors(self, counts):
        vectors = []

        for c in counts:
            vector = []
            for key in self.all_keys:
                if key in c:
                    vector.append(c[key])
                else:
                    vector.append(0)
            vectors.append(np.array(vector))

        return np.array(vectors)

    def vectorise_dataset(self, graphs):
        pass

    def vectorise_dataset_with_keys(self, graphs):
        pass

    def fit_transform(self, X):
        self.phi_x = np.array(self.vectorise_dataset(X))

        km = self.phi_x.dot(self.phi_x.T)

        return km

    def transform(self, X):
        self.phi_y = np.array(self.vectorise_dataset_with_keys(X))

        km = self.phi_y.dot(self.phi_x.T)

        return km

    def get_train_test_sets(self, G_train, G_test, normalize=True):
        # print("\t\tConverting graphs to networkx...")
        # G_train = self.grakel_to_networkx(G_train)
        # G_test = self.grakel_to_networkx(G_test)
        # print("\t\tDone.")

        start_time = time.time()
        training_sets = self.fit_transform(G_train)
        test_sets = self.transform(G_test)
        time_needed = time.time() - start_time

        if normalize:
            training_sets, test_sets = self.min_max_normalize(training_sets, test_sets)
            return training_sets, test_sets, time_needed

        return training_sets, test_sets, time_needed
