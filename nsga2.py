import os
import math
import random
import argparse
import datetime
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from functools import partial

from numpy.random import sample
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from kernels.Utility import Utility
from Individual import Individual

np.random.seed(42)


def create_and_evaluate(_, model, params, G_train, G_valid, y_train, y_valid, vectorised):
    # Create a random Individual
    g_subset, s_subset = Utility.get_random_graplet_and_vertex_symmetry_subset()
    I = Individual(g_subset, s_subset)

    # Evaluate using your GA fitness function
    I.evaluate(model, params, G_train, G_valid, y_train, y_valid, vectorised)

    return I


def parallel_initialize_population(
        pop_size,
        model,
        params,
        G_train,
        G_valid,
        y_train,
        y_valid,
        vectorised,
        workers=12
    ):

    worker_fn = partial(
        create_and_evaluate,
        model=model,
        params=params,
        G_train=G_train,
        G_valid=G_valid,
        y_train=y_train,
        y_valid=y_valid,
        vectorised=vectorised
    )

    # Use all CPUs or specify number of workers
    with mp.Pool(processes=workers) as pool:
        population = pool.map(worker_fn, range(pop_size))

    return population


# First function to optimize
def accuracy(individual: Individual):
    return individual.accuracy


def f1_score(individual: Individual):
    return individual.f1_score


# Second function to optimize
def fidelity(individual: Individual):
    return individual.fidelity


def explanation_precision(individual: Individual):
    return individual.xai_precision


# Function to find the index of a value in a list
def index_of(a, list):
    try:
        return list.index(a)
    except ValueError:
        return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    values_copy = values[:]
    while len(sorted_list) != len(list1):
        min_index = index_of(min(values_copy), values_copy)
        if min_index in list1:
            sorted_list.append(min_index)
        values_copy[min_index] = math.inf
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(len(values1))]
    rank = [0 for _ in range(len(values1))]

    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or \
               (values1[p] >= values1[q] and values2[p] > values2[q]) or \
               (values1[p] > values1[q] and values2[p] >= values2[q]):
                S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or \
                 (values1[q] >= values1[p] and values2[q] > values2[p]) or \
                 (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    del front[-1]
    return front


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for _ in range(len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = distance[-1] = float('inf')
    for k in range(1, len(front) - 1):
        distance[k] += (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1) + 0.000000001)
        distance[k] += (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2) + 0.000000001)
    return distance


def crossover(solution1: Individual, solution2: Individual):
    g1 = solution1.g_subset
    g2 = solution2.g_subset

    s1 = solution1.s_subset
    s2 = solution2.s_subset

    def uniform_set_crossover(set1, set2, p_keep=0.5):
        s1, s2 = set(set1), set(set2)
        union = s1.union(s2)

        child = {elem for elem in union if random.random() < p_keep}
        return list(child)

    new_g = uniform_set_crossover(g1, g2)
    new_s = uniform_set_crossover(s1, s2)

    return Individual(new_g, new_s)


def mutation(solution: Individual):
    graphlets = solution.g_subset
    symmetries = solution.s_subset

    def modify_genome(subset, full_set, p_add=0.2, p_remove=0.2, p_replace=0.1):
        set_diff = list(set(full_set) - set(subset))
        if random.random() < p_add:
            subset.append(random.choice(set_diff))  # add random graphlet
        if subset and random.random() < p_remove:
            subset.remove(random.choice(subset))
        if random.random() < p_replace:
            if subset:
                i = random.randrange(len(subset))
                subset[i] = random.choice(set_diff)

    modify_genome(graphlets, Utility.get_graphlet_index())
    modify_genome(symmetries, Utility.get_symmetry_index())

    return Individual(graphlets, symmetries)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", help="dataset", default="MUTAG", type=str,
                        choices=["AIDS", "BBBP", "clintox", "Mutagenicity", "NCI1", "NCI109", "PROTEINS", "BZR",
                                 "COX2", "DHFR", "MUTAG", "PTC_FM", "PTC_FR", "PTC_MM", "OHSU", "REDDIT-BINARY",
                                 "IMDB-BINARY", "github_stargazers"])

    parser.add_argument("-m", "--model", help="select ML model", default="svc", type=str)
    parser.add_argument("-g", "--graphs", help="graph dataset or loaded x", action="store_true")
    parser.add_argument("-i", "--index", help="vectorised sample index", default=0, type=int)

    return parser


def main():
    # os.chdir("/home/yannick/FRI/DR/advanced-graph-kernels/optimization")
    # os.chdir("/home/yannick/FRI/agk-docker")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    sample_index = args.index

    svc_params = {"C": [1, 5], "kernel": ["rbf"], "gamma": ("scale", "auto")}
    rf_params = {"n_estimators": [50, 100], "criterion": ("gini", "entropy"), "max_depth": [10, 50]}
    ada_params = {"n_estimators": [50, 100], "learning_rate": [0.1, 0.5, 1], }

    if args.model == "svc":
        model, params = (SVC, svc_params)
    elif args.model == "rf":
        model, params = (RandomForestClassifier, rf_params)
    elif args.model == "ada":
        model, params = (AdaBoostClassifier, ada_params)
    else:
        print("[WARNING] Invalid model, default selected is KNN.")
        model, params = (SVC, svc_params)

    # print(dataset, args.model, f"graphs: {args.graphs}")

    if args.graphs:
        graph_labels_file = f"../data/{dataset}/{dataset}_graph_labels.txt"
        graph_edges_file = f"../data/{dataset}/{dataset}_A.txt"
        graph_indicator_file = f"../data/{dataset}/{dataset}_graph_indicator.txt"

        G, y = Utility.parse_dataset_to_nx(graph_labels_file, graph_edges_file, graph_indicator_file)
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3)
        G_train, G_valid, y_train, y_valid = train_test_split(G_train, y_train, test_size=0.3)
    else:
        file_x = f"vectorised_and_reduced_datasets/sample_{sample_index}_{dataset}.csv"
        file_y = f"vectorised_and_reduced_datasets/sample_{sample_index}_{dataset}_classes.csv"
        G, y = Utility.read_csv(file_x, file_y)

        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3)
        G_train, G_valid, y_train, y_valid = train_test_split(G_train, y_train, test_size=0.3)

    pop_size = 1
    max_gen =0
    vectorised = not args.graphs

    # Initialization
    solution = parallel_initialize_population(
        pop_size=30,
        model=model,  # scikit-learn SVC
        params=params,
        G_train=G_train,
        G_valid=G_valid,
        y_train=y_train,
        y_valid=y_valid,
        vectorised=vectorised
    )

    gen_no = 0

    function1 = accuracy
    function2 = fidelity

    while gen_no < max_gen:
        function1_values = [function1(solution[i]) for i in range(pop_size)]
        function2_values = [function2(solution[i]) for i in range(pop_size)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        # Store progress for visualization
        # progress.append((function1_values, function2_values))

        crowding_distance_values = []
        for i in range(len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]

        # Generating offsprings
        while len(solution2) != 2 * pop_size:
            a1 = random.randint(0, pop_size - 1)
            b1 = random.randint(0, pop_size - 1)

            new_sol = crossover(solution[a1], solution[b1])
            new_sol = mutation(new_sol)
            new_sol.evaluate(model, params, G_train, G_valid, y_train, y_valid, vectorised)

            solution2.append(new_sol)

        function1_values2 = [function1(solution2[i]) for i in range(2 * pop_size)]
        function2_values2 = [function2(solution2[i]) for i in range(2 * pop_size)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        new_solution = []
        for i in range(len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == pop_size:
                    break
            if len(new_solution) == pop_size:
                break
        solution = [solution2[i] for i in new_solution]
        gen_no += 1
        print("Generation number:", gen_no)

    print("---------------")

    name = f"result_{model.__name__}_{dataset}_sample_{sample_index}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_log.txt"

    with open(f"logs/{name}", "w") as f:
        for s in solution:
            f.write(f"Graphlet subset: {s.g_subset}\n")
            f.write(f"Symmetry Features: {s.s_subset}\n")
            f.write(f"Graphlet subset: {s.g_subset}\n")
            f.write(f"Symmetry Features: {s.s_subset}\n")
            f.write(f"Importances: {s.importances}\n")

            accs = []
            fidelties = []

            for _ in range(30):
                G_test_sample, y_test_sample = Utility.boostrap_sample_graphs(G_test, y_test)
                baseline_acc = s.test_accuracy(G_test_sample, y_test_sample, vectorised=not args.graphs)
                test_fidelity = s.test_fidelity(baseline_acc)

                accs.append(baseline_acc)
                fidelties.append(test_fidelity)

            accs = np.array(accs)
            fidelties = np.array(fidelties)

            final_acc = round(np.mean(accs), 4)
            final_acc_err = round(np.std(accs), 4)

            final_fidelity = round(np.mean(fidelties), 4)
            final_fidelity_err = round(np.std(fidelties), 4)

            f.write(f"Accuracy: {final_acc}\n")
            f.write(f"Accuracy Error: {final_acc_err}\n")
            # f.write(f"XAI Precision: {s.xai_precision}")
            f.write(f"Fidelity: {final_fidelity}\n")
            f.write(f"Fidelity Error: {final_fidelity_err}\n")
            f.write("-" * 10 + "\n")
            # s.store_classifier(name)


if __name__ == "__main__":
    main()

