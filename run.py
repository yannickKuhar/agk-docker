import os
import time
import shap
import datetime
import argparse

import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from kernels.Utility import Utility
from kernels.GraphletKernel import GraphletKernel
from kernels.EdgeSymmetryKernel import EdgeSymmetryKernel
from kernels.VertexSymmetryKernel import VertexSymmetryKernel
from kernels.EdgeSymmetryKernelOnes import EdgeSymmetryKernelOnes
from kernels.VertexSymmetryKernelOnes import VertexSymmetryKernelOnes
from kernels.GeneralVertexSymmetryKernel import GeneralVertexSymmetryKernelOnes

# Random seed.
np.random.seed(42)


def explanation_precision(importances, k=3):
    return np.sum(np.sort(importances)[k:])


def get_fidelity(baseline_acc, model, phi_train, y_train, phi_test, y_test, best_params, importances, k=3):
    top_features = sorted(importances)[k:]
    indices = np.where(np.isin(importances, top_features))[0]

    phi_train[:, indices] = 0

    clf = model(**best_params)
    clf.fit(phi_train, y_train)

    y_pred = clf.predict(phi_test)
    acc = accuracy_score(y_test, y_pred)

    return baseline_acc - acc


def shap_explain(clf, phi_train, sample_size_background=100, sample_size_explain=100):
    background_sample = shap.sample(phi_train, sample_size_background)

    explainer = shap.KernelExplainer(clf.predict, background_sample)

    explain_sample = shap.sample(phi_train, sample_size_explain)

    shap_values = explainer(explain_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    mean_abs_shap /= (mean_abs_shap.sum() + 0.00000001)

    return mean_abs_shap


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, model, params, n_models=30):
    # (2.) Grid search for hyperparameter optimisation.
    print("\t\tSearching for best hyperparameters...")
    predictor = model()
    cv_model = GridSearchCV(predictor, params)
    cv_model.fit(X_train, y_train)
    best_params = cv_model.best_params_
    print(f"\t\tBest hyperparameters: {cv_model.best_params_}")

    # (3.) Train model with the best hyperparameters.
    f1s = []
    accs = []
    times = []
    # xai_precisions = []
    fidelities = []

    for i in range(n_models):
        print(f"\t\tTraining {predictor.__class__.__name__} iteration {i}...")
        X_train_sample, y_train_sample = Utility.bootsrap_sample(X_train, y_train)

        clf = model(**best_params)

        start_time = time.time()
        clf.fit(X_train_sample, y_train_sample)
        time_needed = time.time() - start_time

        importances = shap_explain(clf, X_train_sample)
        # xai_precisions.append(explanation_precision(importances))

        y_pred = clf.predict(X_test)
        ca_score = accuracy_score(y_test, y_pred)

        fidelities.append(get_fidelity(ca_score, model, X_train_sample, y_train_sample, X_test, y_test,
                                       best_params, importances))

        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accs.append(ca_score)
        times.append(time_needed)

    f1s = np.array(f1s)
    accs = np.array(accs)
    times = np.array(times)
    # xai_precisions = np.array(xai_precisions)
    fidelities = np.array(fidelities)

    return round(np.mean(times), 4), round(np.std(times), 4), round(np.mean(accs), 4), round(np.std(accs), 4),\
           round(np.mean(f1s), 4), round(np.std(f1s), 4), round(np.mean(fidelities), 4), round(np.std(fidelities), 4)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", default="AIDS", help="Dataset to use", choices=["AIDS",
                                    "Mutagenicity", "NCI1", "NCI109", "PROTEINS", "BZR", "COX2", "DHFR", "MUTAG",
                                    "PTC_FM", "PTC_FR", "PTC_MM", "OHSU", "REDDIT-BINARY", "IMDB-BINARY",
                                    "github_stargazers"])

    parser.add_argument("-m", "--model", help="select ML model", default="knn",
                        type=str, choices=["svc", "rf", "ada", "lr", "nb", "knn"])
    parser.add_argument("-n", "--name", help="name of the kernel", default="kernel", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel", default="esko",
                        type=str, choices=["esko", "vsko", "esk", "vsk", "gk", "wl", "general"])

    parser.add_argument("-p", "--phi", help="will phi vectors be used or gram matrix",
                        action="store_true")

    return parser


def main():
    # os.chdir("/home/yannick/FRI/agk-docker")

    svc_params = {"C": [1, 5], "kernel": ["rbf"], "gamma": ("scale", "auto")}
    rf_params = {"n_estimators": [50, 100], "criterion": ("gini", "entropy"), "max_depth": [10, 50]}
    ada_params = {"n_estimators": [50, 100], "learning_rate": [0.1, 0.5, 1]}

    lr_params = {"C": [1, 5, 10], "solver": ("newton-cg", "lbfgs", "liblinear", "sag", "saga")}

    nb_params = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

    knn_params = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ("uniform", "distance")}

    parser = get_parser()
    args = parser.parse_args()

    k = None
    results = []

    if args.kernel == "esko":
        k = EdgeSymmetryKernelOnes()
    elif args.kernel == "vsko":
        k = VertexSymmetryKernelOnes()
    elif args.kernel == "esk":
        k = EdgeSymmetryKernel()
    elif args.kernel == "vsk":
        k = VertexSymmetryKernel()
    elif args.kernel == "gk":
        k = GraphletKernel()
    elif args.kernel == "general":
        g_subset, s_subset = Utility.get_ransom_graplet_and_vertex_symmetry_subset()

        row1 = ["Graphlet subset"]
        row2 = ["Symmetry subset"]

        row1.extend(g_subset)
        row2.extend(s_subset)

        results.append(row1)
        results.append(row2)

        k = GeneralVertexSymmetryKernelOnes(g_subset, s_subset)
    else:
        print("[WARNING] Invalid kernel, default selected is ESKO.")
        k = EdgeSymmetryKernelOnes()

    if args.model == "svc":
        model, params = (SVC, svc_params)
    elif args.model == "rf":
        model, params = (RandomForestClassifier, rf_params)
    elif args.model == "ada":
        model, params = (AdaBoostClassifier, ada_params)
    elif args.model == "lr":
        model, params = (LogisticRegression, lr_params)
    elif args.model == "nb":
        model, params = (GaussianNB, nb_params)
    elif args.model == "knn":
        model, params = (KNeighborsClassifier, knn_params)
    else:
        print("[WARNING] Invalid model, default selected is KNN.")
        model, params = (KNeighborsClassifier, knn_params)

    dataset = args.dataset

    print(f"Training {model.__name__} on dataset {dataset}...")
    row = [dataset]

    print("\t\tParsing dataset...")
    graph_labels_file = f"./data/{dataset}/{dataset}_graph_labels.txt"
    graph_edges_file = f"./data/{dataset}/{dataset}_A.txt"
    graph_indicator_file = f"./data/{dataset}/{dataset}_graph_indicator.txt"

    if args.phi:
        G, y = Utility.parse_dataset_to_nx(graph_labels_file, graph_edges_file, graph_indicator_file)
    else:
        G, y = Utility.parse_dataset(graph_labels_file, graph_edges_file, graph_indicator_file)

    print("\t\tParsing done.")

    print("\t\tCalculating Train test split...")
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3)

    if args.kernel == "wl":
        K_train = k.fit_transform(G_train)
        K_test = k.transform(G_test)

        mc = Utility.get_mc(G, y)
        print(f"\t\tMC Score: {mc}")

        row.append(mc)
        row.extend(
            train_and_evaluate_classifier(K_train, K_test, np.array(y_train), np.array(y_test), model, params))

        results.append(row)
    else:
        if args.phi:
            G_train = k.vectorise_dataset(G_train)
            G_test = k.vectorise_dataset_with_keys(G_test)
        else:
            G_train, G_test, fitting_time = k.get_train_test_sets(G_train, G_test)

        mc = Utility.get_mc(G, y)
        print(f"\t\tMC Score: {mc}")

        row.append(mc)
        row.extend(
            train_and_evaluate_classifier(G_train, G_test, np.array(y_train), np.array(y_test), model, params))

        results.append(row)

    # start_time = time.time()
    # G_train, G_test, fitting_time = k.get_train_test_sets(G_train, G_test)
    # print("\t\tTrain test split done.")
    #
    # with open("fitting_times.txt", "a") as f:
    #     f.write(f"{dataset}, {k.__class__.__name__}, {fitting_time}\n")

    print(results)

    path = f"results/{args.name}_{model.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.mkdir(path)
    Utility.save_results(path, results, args.name)


if __name__ == "__main__":
    main()
