import numpy as np
import networkx as nx

from itertools import combinations
from collections import defaultdict


class WeisfeilerLehmanKernel:

    def __init__(self):
        pass

    def make_mtx(self, fv_dicts, keys):
        mtx = []

        fv_to_i = {k:i for i, k in enumerate(keys)}

        n_graphs = len(fv_dicts[0])

        for i in range(n_graphs):
            graph_fv = [0 for _ in range(len(keys))]

            for dict_list in fv_dicts:
                for key in keys:
                    graph_fv[fv_to_i[key]] += dict_list[i][key]

            mtx.append(graph_fv)

        return mtx

    def vectorise_test_set(self, fv_train, fv_test, label_dict_train, label_dict_test):
        common_labels = list(set(label_dict_train.keys()) & set(label_dict_test))
        fv_train_labels = []
        fv_test_labels = []

        for k in common_labels:
            fv_train_labels.append(label_dict_train[k])
            fv_test_labels.append(label_dict_test[k])

        mtx_train = np.array(self.make_mtx(fv_train, common_labels))
        mtx_test = np.array(self.make_mtx(fv_test, common_labels))

        x_test = np.matmul(mtx_test, mtx_train.T)

        return x_test

    def vectorise_set(self, fv_dicts):
        dataset = []
        keys = []

        for dict_list in fv_dicts:
            for dict in dict_list:
                keys += dict.keys()

        keys = sorted(keys)

        n_graphs = len(fv_dicts[0])

        for i in range(n_graphs):
            graph_fv = [0 for _ in range(len(keys))]

            for dict_list in fv_dicts:
                for key in dict_list[i]:
                    graph_fv[key] += dict_list[i][key]

            dataset.append(graph_fv)

        return dataset, keys

    def wl_kernel(self, graphs, h=1):
        all_fv = []

        # Initialize the node labels
        labels = []
        for graph in graphs:
            labels.append({node: str(node) for node in graph})

        # Initialize label dictionary
        label_dict = defaultdict(int)
        rev_label_dict = defaultdict(int)
        label_counter = 0

        for i in range(h):
            # Create the feature vectors for each graph
            feature_vectors = []
            for graph_idx, graph in enumerate(graphs):
                feature_vector = defaultdict(int)

                for node in graph:
                    node_label = labels[graph_idx][node]
                    feature_vector[int(node_label)] += 1

                feature_vectors.append(feature_vector)

            all_fv.append(feature_vectors)

            # Re-label the nodes
            new_labels = []
            for graph_idx, graph in enumerate(graphs):
                new_labeling = {}
                for node in graph:
                    neighborhood_label = labels[graph_idx][node]
                    neighborhood = sorted([labels[graph_idx][neighbor] for neighbor in graph[node]])

                    new_label = str(neighborhood_label) + "_" + "_".join(list(map(str, neighborhood)))

                    if new_label not in label_dict:
                        label_dict[new_label] = label_counter
                        rev_label_dict[label_counter] = new_label
                        label_counter += 1

                    new_labeling[node] = label_dict[new_label]

                new_labels.append(new_labeling)

            labels = new_labels

        return all_fv, label_dict, rev_label_dict
