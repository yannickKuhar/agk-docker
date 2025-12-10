from BaseKernel import BaseKernel


class GraphPartitioningKernel(BaseKernel):
    def __init__(self, graph_partitioning_method):
        BaseKernel.__init__(self)
        self.graph_partitioning_method = graph_partitioning_method

    def get_dist(self, G):
        lens = []

        P = self.graph_partitioning_method(G)

        for p in P:
            lens.append(len(p))

        dist = {}

        for l in lens:
            if l in dist:
                dist[l] += 1
            else:
                dist[l] = 1

        return dist

    def vectorise_dataset(self, graphs):
        counts = []
        all_keys = set()

        for g in graphs:
            dist = self.get_dist(g)

            for key in dist:
                all_keys.add(key)

            counts.append(dist)

        self.all_keys = list(all_keys)

        return self.make_vectors(counts)

    def vectorise_dataset_with_keys(self, graphs):
        if self.all_keys is None:
            print("Please call vectorise_dataset first!")
            exit(-1)

        counts = []

        for g in graphs:
            dist = self.get_dist(g)
            counts.append(dist)

        return self.make_vectors(counts)


