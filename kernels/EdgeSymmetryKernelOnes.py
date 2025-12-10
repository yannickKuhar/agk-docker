import orcapy.orca as orca
from kernels.VertexSymmetryKernel import VertexSymmetryKernel


class EdgeSymmetryKernelOnes(VertexSymmetryKernel):
    def __init__(self):
        VertexSymmetryKernel.__init__(self)
        self.all_keys = ["1,1,2", "1,1,2,1,2,2", "1,1,2,2", "1,2", "1,2,1,2,1,2,2", "1,2,1,2,2", "1,2,2", "1,2,2,2",
                         "1,3", "1,3,3,3", "2", "2,1", "2,1,1,2,2", "2,1,1,2,2,2", "2,1,1,2,2,2,1,2", "2,1,1,2,2,2,2",
                         "2,1,2,1", "2,1,2,1,2", "2,1,2,1,2,2", "2,1,2,2", "2,2", "2,2,1", "2,2,1,1", "2,2,2",
                         "2,2,2,1", "2,2,2,1,1", "2,2,2,1,2", "2,2,2,2", "2,2,2,2,1,1", "2,2,2,2,1,1,2,1,2",
                         "2,2,2,2,1,2,1,2", "2,2,2,2,2", "2,2,2,2,2,1,1", "2,2,2,2,2,2", "2,2,2,2,2,2,2",
                         "2,2,2,2,2,2,2,2", "2,2,2,2,2,2,2,2,2", "2,6,3", "3", "3,1,3,3,3", "3,1,3,3,3,3", "3,3",
                         "3,3,3", "3,3,3,3", "3,3,3,3,1,3", "3,3,3,3,1,3,3", "3,3,3,3,3", "3,3,3,3,3,1", "3,3,6",
                         "3,6,3,2,6", "3,6,3,6", "4", "4,4", "4,4,4", "4,4,4,4", "4,4,4,4,4", "5,5", "5,5,5", "5,5,5,5",
                         "6", "6,2,3,3,6", "6,2,3,6", "6,2,6,3", "6,3", "6,3,3,2,6", "6,3,3,6"]

    def get_edge_symms_orca(self, orbit_counts):
        symms = {x: 0 for x in self.all_keys}

        gs = self.get_graphlet_counts(orbit_counts)

        # Graphlet 1
        symms["2"] += 1 * gs[0]

        # Graphlet 2
        symms["2"] += 1 * gs[1]
        symms["2,2"] += 2 * gs[1]
        symms["3"] += 2 * gs[1]

        # Graphlet 3
        symms["1,2"] += 1 * gs[2]
        symms["2"] += 2 * gs[2]
        symms["3"] += 2 * gs[2]

        # Graphlet 4
        symms["2,2"] += 1 * gs[3]

        # Graphlet 5
        symms["2,2"] += 1 * gs[4]

        # Graphlet 6
        symms["2,2,2"] += 2 * gs[5]
        symms["2,2"] += 3 * gs[5]
        symms["4,4"] += 2 * gs[5]

        # Graphlet 7
        symms["2,2"] += 3 * gs[6]

        # Graphlet 8
        symms["2,2,2"] += 2 * gs[7]
        symms["1,2,2"] += 1 * gs[7]
        symms["2,2,2,1,2"] += 1 * gs[7]
        symms["3,3,3,3"] += 2 * gs[7]
        symms["4,4"] += 2 * gs[7]
        symms["2,1,2,2"] += 2 * gs[7]
        symms["2,2,2,2"] += 2 * gs[7]
        symms["2,2,2,2,2"] += 1 * gs[7]
        symms["3,3,3"] += 4 * gs[7]
        symms["3,3"] += 2 * gs[7]
        symms["4,4,4"] += 4 * gs[7]

        # Graphlet 9
        symms["2,2"] += 3 * gs[8]
        symms["2,1"] += 3 * gs[8]
        symms["3"] += 6 * gs[8]
        symms["1,1,2"] += 1 * gs[8]
        symms["1,3"] += 2 * gs[8]
        symms["1,2"] += 2 * gs[8]
        symms["4"] += 6 * gs[8]

        # Graphlet 10
        symms["1,1,2"] += 1 * gs[9]

        # Graphlet 11
        symms["1,1,2"] += 1 * gs[10]
        symms["2,2,2"] += 1 * gs[10]
        symms["2,2,1"] += 1 * gs[10]

        # Graphlet 12
        symms["2,2"] += 1 * gs[11]

        # Graphlet 13
        symms["2,2"] += 1 * gs[12]

        # Graphlet 14
        symms["2,2,1"] += 1 * gs[13]

        # Graphlet 15
        symms["2,2,2"] += 4 * gs[14]
        symms["1,1,2,2"] += 1 * gs[14]
        symms["6"] += 2 * gs[14]
        symms["2,2,1"] += 2 * gs[14]
        symms["3,3"] += 2 * gs[14]

        # Graphlet 16
        symms["2,2,2"] += 4 * gs[15]
        symms["1,1,2,2"] += 1 * gs[15]
        symms["6"] += 2 * gs[15]
        symms["2,2,1,1"] += 2 * gs[15]
        symms["3,3"] += 2 * gs[15]

        # Graphlet 17
        symms["2,2"] += 1 * gs[16]

        # Graphlet 18
        symms["2,2,1"] += 1 * gs[17]

        # Graphlet 19
        symms["2,2,2"] += 1 * gs[18]
        symms["1,1,2,2"] += 1 * gs[18]
        symms["4,4"] += 2 * gs[18]
        symms["2,2,2,2"] += 2 * gs[18]
        symms["2,2,1,1"] += 1 * gs[18]

        # Graphlet 20
        symms["5,5"] += 4 * gs[19]
        symms["2,2,2"] += 2 * gs[19]
        symms["2,2"] += 1 * gs[19]
        symms["2,2,2,2"] += 2 * gs[19]

        # Graphlet 21
        symms["2,2,2,2,2"] += 1 * gs[20]

        # Graphlet 22
        symms["2,2,2,2,2"] += 1 * gs[21]

        # Graphlet 23
        symms["2,2,2"] += 1 * gs[22]

        # Graphlet 24
        symms["2,2,2,1"] += 1 * gs[23]
        symms["2,1,1,2,2"] += 1 * gs[23]
        symms["3,3,3"] += 2 * gs[23]
        symms["2,2,2,1,2"] += 1 * gs[23]

        # Graphlet 25
        symms["2,2,2,1,1"] += 1 * gs[24]
        symms["2,2,2,2"] += 1 * gs[24]
        symms["1,2,2,2"] += 1 * gs[24]

        # Graphlet 26
        symms["2,2,2"] += 2 * gs[25]
        symms["2,1,2,1"] += 1 * gs[25]

        # Graphlet 27
        symms["2,2,2,1"] += 1 * gs[26]
        symms["2,1,2,1,2"] += 1 * gs[26]
        symms["2,2,2,2,2,2"] += 2 * gs[26]
        symms["2,2,2,2"] += 1 * gs[26]
        symms["4,4,4"] += 2 * gs[26]

        # Graphlet 28
        symms["2,1,2,1,2,2"] += 2 * gs[27]
        symms["6,3"] += 2 * gs[27]
        symms["1,2,1,2,2"] += 1 * gs[27]
        symms["2,2,2,2"] += 1 * gs[27]
        symms["2,2,2,2,2"] += 2 * gs[27]
        symms["3,3,3"] += 2 * gs[27]
        symms["2,2,2,1,1"] += 1 * gs[27]

        # Graphlet 29
        symms["5,5,5,5"] += 20 * gs[28]
        symms["3,1,3,3,3,3"] += 2 * gs[28]
        symms["2,2,2,2,2,2,2,2,2"] += 1 * gs[28]
        symms["3,3,6"] += 2 * gs[28]
        symms["2,2,2,2,2,1,1"] += 1 * gs[28]
        symms["4,4,4,4,4"] += 12 * gs[28]
        symms["5,5,5"] += 4 * gs[28]
        symms["2,2,2,2,2"] += 2 * gs[28]
        symms["3,1,3,3,3"] += 4 * gs[28]
        symms["3,3,3,3"] += 2 * gs[28]
        symms["3,6,3,6"] += 2 * gs[28]
        symms["3,3,3,3,3,1"] += 2 * gs[28]
        symms["1,2,1,2,1,2,2"] += 2 * gs[28]
        symms["3,6,3,2,6"] += 2 * gs[28]
        symms["4,4,4"] += 2 * gs[28]
        symms["2,6,3"] += 2 * gs[28]
        symms["3,3,3,3,1,3"] += 2 * gs[28]
        symms["2,2,2,2,1,1"] += 1 * gs[28]
        symms["3,3,3,3,3"] += 4 * gs[28]
        symms["2,2,2,2,2,2,2,2"] += 3 * gs[28]
        symms["2,2,2,2,1,1,2,1,2"] += 1 * gs[28]
        symms["6,3,3,2,6"] += 2 * gs[28]
        symms["2,1,1,2,2,2,1,2"] += 1 * gs[28]
        symms["6,2,3,3,6"] += 2 * gs[28]
        symms["2,2,2,2,1,2,1,2"] += 1 * gs[28]
        symms["2,2,2,2,2,2,2"] += 5 * gs[28]
        symms["1,3,3,3"] += 2 * gs[28]
        symms["6,2,3,6"] += 2 * gs[28]
        symms["2,1,1,2,2,2"] += 1 * gs[28]
        symms["6,3,3,6"] += 4 * gs[28]
        symms["2,2,2,2,2,2"] += 4 * gs[28]
        symms["2,1,1,2,2,2,2"] += 1 * gs[28]
        symms["1,1,2,1,2,2"] += 1 * gs[28]
        symms["6,2,6,3"] += 2 * gs[28]
        symms["3,3,3,3,1,3,3"] += 2 * gs[28]
        symms["4,4,4,4"] += 16 * gs[28]

        return symms

    def vectorise_dataset(self, graphs):
        counts = []

        print("\t\tVectorising dataset ORCA...")

        i = 1
        l = len(graphs)
        for g in graphs:
            # print(f"\t\t{i}/{l}")
            orbit_counts = orca.orbit_counts("node", 5, g)
            counts.append(self.get_edge_symms_orca(orbit_counts))
            i += 1

        return self.make_vectors(counts)

    vectorise_dataset_with_keys = vectorise_dataset
