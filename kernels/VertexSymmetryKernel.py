import orcapy.orca as orca
from kernels.BaseKernel import BaseKernel


class VertexSymmetryKernel(BaseKernel):
    def __init__(self):
        BaseKernel.__init__(self)
        self.all_keys = ["2", "2,2", "2,3", "3", "3,2", "4", "5"]

    def get_graphlet_counts(self, orbit_counts):
        gs = [0 for _ in range(29)]

        for c in orbit_counts:
            # 3 nodes
            gs[0] += (c[1] + c[2]) / 3
            gs[1] += c[3] / 3

            # 4 nodes
            gs[2] += (c[4] + c[5]) / 4
            gs[3] += (c[6] + c[7]) / 4
            gs[4] += c[8] / 4
            gs[5] += (c[9] + c[10] + c[11]) / 4
            gs[6] += (c[12] + c[13]) / 4
            gs[7] += c[14] / 4

            # 5 nodes
            gs[8] += (c[15] + c[16] + c[17]) / 5
            gs[9] += (c[18] + c[19] + c[20] + c[21]) / 5
            gs[10] += (c[22] + c[23]) / 5
            gs[11] += (c[24] + c[25] + c[26]) / 5
            gs[12] += (c[27] + c[28] + c[29] + c[30]) / 5
            gs[13] += (c[31] + c[32] + c[33]) / 5
            gs[14] += c[34] / 5
            gs[15] += (c[35] + c[36] + c[37] + c[38]) / 5
            gs[16] += (c[39] + c[40] + c[41] + c[42]) / 5
            gs[17] += (c[43] + c[44]) / 5
            gs[18] += (c[45] + c[46] + c[47] + c[48]) / 5
            gs[19] += (c[49] + c[50]) / 5
            gs[20] += (c[51] + c[52] + c[53]) / 5
            gs[21] += (c[54] + c[55]) / 5
            gs[22] += (c[56] + c[57] + c[58]) / 5
            gs[23] += (c[59] + c[60] + c[61]) / 5
            gs[24] += (c[62] + c[63] + c[64]) / 5
            gs[25] += (c[65] + c[66] + c[67]) / 5
            gs[26] += (c[68] + c[69]) / 5
            gs[27] += (c[70] + c[71]) / 5
            gs[28] += c[72] / 5

        return list(map(lambda x: int(round(x)), gs))

    def get_symms_orca(self, orbit_counts):
        symms = {k: 0 for k in self.all_keys}

        gs = self.get_graphlet_counts(orbit_counts)

        # Graphlet 1
        symms["2"] += gs[0]

        # Graphlet 2
        symms["2"] += 3 * gs[1]
        symms["3"] = 2 * gs[1]

        # Graphlet 3
        symms["2,2"] += gs[2]

        # Graphlet 4
        symms["2"] += 3 * gs[3]
        symms["3"] = 2 * gs[3]

        # Graphlet 5
        symms["2"] += 2 * gs[4]
        symms["2,2"] += 3 * gs[4]
        symms["4"] += 2 * gs[4]

        # Graphlet 6
        symms["2"] += gs[5]

        # Graphlet 7
        symms["2"] += 2 * gs[6]
        symms["2,2"] += gs[6]

        # Graphlet 8
        symms["2"] += 6 * gs[7]
        symms["2,2"] += 3 * gs[7]
        symms["3"] += 8 * gs[7]
        symms["4"] += 6 * gs[7]

        # Graphlet 9
        symms["2,2"] += gs[8]

        # Graphlet 10
        symms["2"] += gs[9]

        # Graphlet 11
        symms["2"] += 6 * gs[10]
        symms["2,2"] += 3 * gs[10]
        symms["3"] += 8 * gs[10]
        symms["4"] += 6 * gs[10]

        # Graphlet 12
        symms["2,2"] += gs[11]

        # Graphlet 13
        symms["2"] += gs[12]

        # Graphlet 14
        symms["2"] += 2 * gs[13]
        symms["2,2"] += 1 * gs[13]

        # Graphlet 15
        symms["2,2"] += 5 * gs[14]
        symms["5"] += 4 * gs[14]

        # Graphlet 16
        symms["2"] += gs[15]

        # Graphlet 17
        symms["2"] += gs[16]

        # Graphlet 18
        symms["2"] += 2 * gs[17]
        symms["2,2"] += 3 * gs[17]
        symms["4"] += 2 * gs[17]

        # Graphlet 19
        symms["2"] += gs[18]

        # Graphlet 20
        symms["2"] += 4 * gs[19]
        symms["2,2"] += 3 * gs[19]
        symms["3"] += 2 * gs[19]
        symms["3,2"] += 2 * gs[19]

        # Graphlet 21
        symms["2,2"] += gs[20]

        # Graphlet 22
        symms["2"] += 4 * gs[21]
        symms["2,2"] += 3 * gs[21]
        symms["3"] += 2 * gs[21]
        symms["3,2"] += 2 * gs[21]

        # Graphlet 23
        symms["2"] += 3 * gs[22]
        symms["3"] += 2 * gs[22]

        # Graphlet 24
        symms["2,2"] += gs[23]

        # Graphlet 25
        symms["2"] += 2 * gs[24]
        symms["2,2"] += gs[24]

        # Graphlet 26
        symms["2"] += 2 * gs[25]
        symms["2,2"] += gs[25]

        # Graphlet 27
        symms["2"] += 2 * gs[26]
        symms["2,2"] += 3 * gs[26]
        symms["4"] += 2 * gs[26]

        # Graphlet 28
        symms["2"] += 4 * gs[27]
        symms["2,2"] += 3 * gs[27]
        symms["2,3"] += 2 * gs[27]
        symms["3"] += 2 * gs[27]

        # Graphlet 29
        symms["2"] += 10 * gs[28]
        symms["2,2"] += 15 * gs[28]
        symms["2,3"] += 8 * gs[28]
        symms["3"] += 20 * gs[28]
        symms["3,2"] += 12 * gs[28]
        symms["4"] += 30 * gs[28]
        symms["5"] += 24 * gs[28]

        return symms

    def vectorise_dataset(self, graphs):
        counts = []

        for g in graphs:
            orbit_counts = orca.orbit_counts("node", 5, g)
            counts.append(self.get_symms_orca(orbit_counts))

        return self.make_vectors(counts)

    vectorise_dataset_with_keys = vectorise_dataset
