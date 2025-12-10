import os


def get_small_data(results_dir):
    g_subsets = []
    s_subsets = []
    tables = []

    files = os.walk(results_dir)
    files = list(files)[1:]

    for dir, _, file in files:
        if len(file) > 0:
            results = dir + "/" + file[0]

            with open(results, "r") as f:
                lines = f.readlines()

                g_subset = list(map(int, lines[1].split(",")[1:]))
                s_subset = list(map(int, lines[2].split(",")[1:]))

                g_subsets.append(g_subset)
                s_subsets.append(s_subset)

                small_table =[]

                for line in lines[3:]:
                    line = list(map(float, line.split(",")[1:]))
                    line = [line[3], line[5]]
                    small_table.append(line)

                tables.append(small_table)
    return files, g_subsets, s_subsets, tables


def find_best(tables, i, j):
    best = -1
    best_score = -1
    for idx, table in enumerate(tables):
        if table[i][j] > best_score:
            best_score = table[i][j]
            best = idx

    return best, best_score


def main():
    results_dir = "GeneralKernelResults"

    datasets = ["AIDS", "Mutagenicity", "NCI1", "NCI109", "PROTEINS", "BZR", "COX2", "DHFR", "MUTAG", "PTC_FM",
                "PTC_FR", "PTC_MM", "OHSU"]

    colls = ["Acc", "F1"]

    files, g_subsets, s_subsets, tables = get_small_data(results_dir)

    for i in range(len(datasets)):
        for j in [0, 1]:
            best, best_score = find_best(tables, i, j)
            print(colls[j], best, best_score)
            print(g_subsets[best])
            print(s_subsets[best])
            print(files[best][0])
            print("-------------")


if __name__ == "__main__":
    main()
