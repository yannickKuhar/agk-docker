import numpy as np
import orcapy.orca as orca
from kernels.VertexSymmetryKernelOnes import VertexSymmetryKernelOnes


class GeneralVertexSymmetryKernelOnes(VertexSymmetryKernelOnes):

    def __init__(self, graphlet_index, symmetry_index):
        VertexSymmetryKernelOnes.__init__(self)
        self.graphlet_index = graphlet_index
        self.symmetry_index = symmetry_index
        self.full_symm_set_keys = self.all_keys

        keys = [self.all_keys[i] for i in symmetry_index]
        self.all_keys = keys

    def get_general_symms(self, orbit_counts):
        symms = {x: 0 for x in self.all_keys}

        gs = self.get_graphlet_counts(orbit_counts)

        for i in range(len(gs)):
            if i not in self.graphlet_index:
                gs[i] = 0

        if "2,1" in self.all_keys:
            symms["2,1"] += gs[0]
            symms["2,1"] += 3 * gs[1]

        if "2,2" in self.all_keys:
            symms["2,2"] += gs[2]
            symms["2,2"] += gs[6]
            symms["2,2"] += 3 * gs[7]

        if "2,3" in self.all_keys:
            symms["2,3"] += 2 * gs[27]
            symms["2,3"] += 8 * gs[28]

        if "2,1,1" in self.all_keys:
            symms["2,1,1"] += 3 * gs[3]
            symms["2,1,1"] += 2 * gs[4]
            symms["2,1,1"] += gs[5]
            symms["2,1,1"] += 2 * gs[6]
            symms["2,1,1"] += 6 * gs[7]

        if "2,2,1" in self.all_keys:
            symms["2,2,1"] += 3 * gs[4]
            symms["2,2,1"] += gs[8]
            symms["2,2,1"] += 3 * gs[10]
            symms["2,2,1"] += gs[11]
            symms["2,2,1"] += 1 * gs[13]
            symms["2,2,1"] += 5 * gs[14]
            symms["2,2,1"] += 3 * gs[17]
            symms["2,2,1"] += 3 * gs[19]
            symms["2,2,1"] += gs[20]
            symms["2,2,1"] += 3 * gs[21]
            symms["2,2,1"] += gs[23]
            symms["2,2,1"] += gs[24]
            symms["2,2,1"] += gs[25]
            symms["2,2,1"] += 3 * gs[26]
            symms["2,2,1"] += 3 * gs[27]
            symms["2,2,1"] += 15 * gs[28]

        if "2,1,1,1" in self.all_keys:
            symms["2,1,1,1"] += gs[9]
            symms["2,1,1,1"] += 6 * gs[10]
            symms["2,1,1,1"] += gs[12]
            symms["2,1,1,1"] += 2 * gs[13]
            symms["2,1,1,1"] += gs[15]
            symms["2,1,1,1"] += gs[16]
            symms["2,1,1,1"] += 2 * gs[17]
            symms["2,1,1,1"] += gs[18]
            symms["2,1,1,1"] += 4 * gs[19]
            symms["2,1,1,1"] += 4 * gs[21]
            symms["2,1,1,1"] += 3 * gs[22]
            symms["2,1,1,1"] += 2 * gs[24]
            symms["2,1,1,1"] += 2 * gs[25]
            symms["2,1,1,1"] += 2 * gs[26]
            symms["2,1,1,1"] += 4 * gs[27]
            symms["2,1,1,1"] += 10 * gs[28]

        if "3" in self.all_keys:
            symms["3"] = 2 * gs[1]

        if "3,1" in self.all_keys:
            symms["3,1"] = 2 * gs[3]
            symms["3,1"] += 8 * gs[7]

        if "3,1,1" in self.all_keys:
            symms["3,1,1"] += 8 * gs[10]
            symms["3,1,1"] += 2 * gs[19]
            symms["3,1,1"] += 2 * gs[21]
            symms["3,1,1"] += 2 * gs[22]
            symms["3,1,1"] += 2 * gs[27]
            symms["3,1,1"] += 20 * gs[28]

        if "3,2" in self.all_keys:
            symms["3,2"] += 2 * gs[19]
            symms["3,2"] += 2 * gs[21]
            symms["3,2"] += 12 * gs[28]

        if "4" in self.all_keys:
            symms["4"] += 2 * gs[4]
            symms["4"] += 6 * gs[7]

        if "4,1" in self.all_keys:
            symms["4,1"] += 6 * gs[10]
            symms["4,1"] += 2 * gs[17]
            symms["4,1"] += 2 * gs[26]
            symms["4,1"] += 30 * gs[28]

        if "5" in self.all_keys:
            symms["5"] += 4 * gs[14]
            symms["5"] += 24 * gs[28]

        result_symms ={}
        for s in self.symmetry_index:
            symm = self.full_symm_set_keys[s]
            result_symms[symm] = symms[symm]

        return result_symms

    def make_vectors_vectorised(self, counts):
        vectors = []
        for c in counts:
            vector = []
            for key in self.symmetry_index:
                vector.append(c[key])

            vectors.append(np.array(vector))

        return np.array(vectors)

    def vectorise_dataset(self, graphs):
        counts = []

        print("\t\tVectorising dataset ORCA...")

        i = 1
        for g in graphs:
            # print(f"\t\t{i}/{l}")
            orbit_counts = orca.orbit_counts("node", 5, g)
            counts.append(self.get_general_symms(orbit_counts))
            i += 1

        return self.make_vectors(counts)

    vectorise_dataset_with_keys = vectorise_dataset
