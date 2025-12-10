import orcapy.orca as orca
from kernels.VertexSymmetryKernel import VertexSymmetryKernel


class VertexSymmetryKernelOnes(VertexSymmetryKernel):
    def __init__(self):
        VertexSymmetryKernel.__init__(self)
        self.all_keys = ["2,1", "3", "2,1,1", "2,2", "3,1", "4", "2,1,1,1", "2,2,1", "2,3", "3,1,1", "3,2", "4,1", "5"]

    def get_symms_with_ones_orca(self, orbit_counts):
        symms = {x: 0 for x in self.all_keys}

        gs = self.get_graphlet_counts(orbit_counts)

        # Graphlet 1
        symms["2,1"] += gs[0]

        # Graphlet 2
        symms["2,1"] += 3 * gs[1]
        symms["3"] = 2 * gs[1]

        # Graphlet 3
        symms["2,2"] += gs[2]

        symms["2,1,1"] += 3 * gs[3]
        symms["3,1"] = 2 * gs[3]

        # Graphlet 5
        symms["2,1,1"] += 2 * gs[4]
        symms["2,2,1"] += 3 * gs[4]
        symms["4"] += 2 * gs[4]

        # Graphlet 6
        symms["2,1,1"] += gs[5]

        # Graphlet 7
        symms["2,1,1"] += 2 * gs[6]
        symms["2,2"] += gs[6]

        # Graphlet 8
        symms["2,1,1"] += 6 * gs[7]
        symms["2,2"] += 3 * gs[7]
        symms["3,1"] += 8 * gs[7]
        symms["4"] += 6 * gs[7]

        # Graphlet 9
        symms["2,2,1"] += gs[8]

        # Graphlet 10
        symms["2,1,1,1"] += gs[9]

        # Graphlet 11
        symms["2,1,1,1"] += 6 * gs[10]
        symms["2,2,1"] += 3 * gs[10]
        symms["3,1,1"] += 8 * gs[10]
        symms["4,1"] += 6 * gs[10]

        # Graphlet 12
        symms["2,2,1"] += gs[11]

        # Graphlet 13
        symms["2,1,1,1"] += gs[12]

        # Graphlet 14
        symms["2,1,1,1"] += 2 * gs[13]
        symms["2,2,1"] += 1 * gs[13]

        # Graphlet 15
        symms["2,2,1"] += 5 * gs[14]
        symms["5"] += 4 * gs[14]

        # Graphlet 16
        symms["2,1,1,1"] += gs[15]

        # Graphlet 17
        symms["2,1,1,1"] += gs[16]

        # Graphlet 18
        symms["2,1,1,1"] += 2 * gs[17]
        symms["2,2,1"] += 3 * gs[17]
        symms["4,1"] += 2 * gs[17]

        # Graphlet 19
        symms["2,1,1,1"] += gs[18]

        # Graphlet 20
        symms["2,1,1,1"] += 4 * gs[19]
        symms["2,2,1"] += 3 * gs[19]
        symms["3,1,1"] += 2 * gs[19]
        symms["3,2"] += 2 * gs[19]

        # Graphlet 21
        symms["2,2,1"] += gs[20]

        # Graphlet 22
        symms["2,1,1,1"] += 4 * gs[21]
        symms["2,2,1"] += 3 * gs[21]
        symms["3,1,1"] += 2 * gs[21]
        symms["3,2"] += 2 * gs[21]

        # Graphlet 23
        symms["2,1,1,1"] += 3 * gs[22]
        symms["3,1,1"] += 2 * gs[22]

        # Graphlet 24
        symms["2,2,1"] += gs[23]

        # Graphlet 25
        symms["2,1,1,1"] += 2 * gs[24]
        symms["2,2,1"] += gs[24]

        # Graphlet 26
        symms["2,1,1,1"] += 2 * gs[25]
        symms["2,2,1"] += gs[25]

        # Graphlet 27
        symms["2,1,1,1"] += 2 * gs[26]
        symms["2,2,1"] += 3 * gs[26]
        symms["4,1"] += 2 * gs[26]

        # Graphlet 28
        symms["2,1,1,1"] += 4 * gs[27]
        symms["2,2,1"] += 3 * gs[27]
        symms["2,3"] += 2 * gs[27]
        symms["3,1,1"] += 2 * gs[27]

        # Graphlet 29
        symms["2,1,1,1"] += 10 * gs[28]
        symms["2,2,1"] += 15 * gs[28]
        symms["2,3"] += 8 * gs[28]
        symms["3,1,1"] += 20 * gs[28]
        symms["3,2"] += 12 * gs[28]
        symms["4,1"] += 30 * gs[28]
        symms["5"] += 24 * gs[28]

        return symms

    def vectorise_dataset(self, graphs):
        counts = []

        print("\t\tVectorising dataset ORCA...")

        i = 1
        for g in graphs:
            # print(f"\t\t{i}/{l}")
            orbit_counts = orca.orbit_counts("node", 5, g)
            counts.append(self.get_symms_with_ones_orca(orbit_counts))
            i += 1

        return self.make_vectors(counts)

    vectorise_dataset_with_keys = vectorise_dataset
