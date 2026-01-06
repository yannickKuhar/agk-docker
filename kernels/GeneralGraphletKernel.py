import orcapy.orca as orca
from kernels.GraphletKernel import GraphletKernel


class GeneralGraphletKernel(GraphletKernel):
    def __init__(self, graphlet_index):
        GraphletKernel.__init__(self)
        self.graphlet_index = graphlet_index

    def vectorise_dataset(self, graphs):
        counts = []
        for g in graphs:
            orbit_counts = orca.orbit_counts("node", 5, g)
            graphlet_counts = self.get_graphlet_counts(orbit_counts)
            general_count = [graphlet_counts[i] for i in self.graphlet_index]
            counts.append(general_count)

        return counts

    vectorise_dataset_with_keys = vectorise_dataset
        