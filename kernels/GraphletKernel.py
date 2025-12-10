import orcapy.orca as orca

from kernels.VertexSymmetryKernel import VertexSymmetryKernel


class GraphletKernel(VertexSymmetryKernel):
    def __init__(self):
        VertexSymmetryKernel.__init__(self)
        self.all_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                         "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]

    def vectorise_dataset(self, graphs):
        counts = []

        for g in graphs:
            orbit_counts = orca.orbit_counts("node", 5, g)
            counts.append(self.get_graphlet_counts(orbit_counts))

        return counts

    vectorise_dataset_with_keys = vectorise_dataset
