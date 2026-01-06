import os
import shap
import pickle
import datetime
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from kernels.Utility import Utility
# from kernels.GeneralVertexSymmetryKernel import GeneralVertexSymmetryKernelOnes
from kernels.GeneralGraphletKernel import GeneralGraphletKernel


class Individual:
    def __init__(self, g_subset, s_subset, k=3):
        self.k = k
        self.g_subset = g_subset
        self.s_subset = s_subset

        self.kernel = GeneralGraphletKernel(self.g_subset)

        self.accuracy = 0
        self.f1_score = 0

        self.importances = None

        self.fidelity = 0
        self.xai_precision = 0

        self.G_train = None
        self.phi_train = None
        self.y_train = None
        self.phi_test = None
        self.y_test = None

        self.best_params = None
        self.model = None
        self.clf = None

    def shap_explain(self, clf, phi_train, sample_size_background=100, sample_size_explain=100):

        if len(phi_train) < sample_size_background:
            background_sample = phi_train
        else:
            background_sample = shap.sample(phi_train, sample_size_background)

        explainer = shap.KernelExplainer(clf.predict, background_sample)

        if len(phi_train) < sample_size_explain:
            explain_sample = phi_train
        else:
            explain_sample = shap.sample(phi_train, sample_size_explain)

        shap_values = explainer(explain_sample)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        mean_abs_shap /= (mean_abs_shap.sum() + 0.00000001)

        return mean_abs_shap

    def explanation_precision(self, importances):
        return np.sum(np.sort(importances)[-self.k:])

    def get_best_params(self, model, params, X_train, y_train):
        predictor = model()
        cv_model = GridSearchCV(predictor, params)
        cv_model.fit(X_train, y_train)
        return cv_model.best_params_

    def train_model(self, model, phi_train, y_train, best_params):
        clf = model(**best_params)
        clf.fit(phi_train, y_train)
        return clf

    def get_fidelity(self, baseline_acc, model, phi_train, y_train, phi_test, y_test, best_params, importances):
        top_features = sorted(importances)[-self.k:]
        indices = np.where(np.isin(importances, top_features))[0]

        phi_train[:, indices] = 0

        clf = self.train_model(model, phi_train, y_train, best_params)
        y_pred = clf.predict(phi_test)
        acc = accuracy_score(y_test, y_pred)

        return baseline_acc - acc

    def store_classifier(self, name):
        path = f"pickle_svm/{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

        with open(path, "ab") as f:
            pickle.dump(self.clf, f)

    def evaluate(self, model, params, G_train, G_valid, y_train, y_valid, vectorised=False):
        # Prepare data
        if vectorised:
            X_train = self.kernel.make_vectors_vectorised(G_train)
            X_valid = self.kernel.make_vectors_vectorised(G_valid)
        else:
            X_train, X_valid, _ = self.kernel.get_train_test_sets(G_train, G_valid)
            X_train = self.kernel.vectorise_dataset(G_train)
            X_valid = self.kernel.vectorise_dataset_with_keys(G_valid)

        # print(X_train)
        train_shape = X_train.shape
        print(train_shape)

        if train_shape[1] > 0:
            self.model = model
            self.G_train = G_train

            self.phi_train = X_train
            self.y_train = y_train

            best_params = self.get_best_params(model, params, X_train, y_train)

            self.best_params = best_params

            clf = self.train_model(model, X_train, y_train, best_params)
            y_pred = clf.predict(X_valid)

            self.clf = clf

            self.f1_score = f1_score(y_valid, y_pred, average="weighted")
            self.accuracy = accuracy_score(y_valid, y_pred)

            self.importances = self.shap_explain(clf, X_train)

            self.fidelity = self.get_fidelity(self.accuracy, model, X_train, y_train, X_valid, y_valid,
                                            best_params, self.importances)

            self.xai_precision = self.explanation_precision(self.importances)

            print(self.accuracy)
        else:
            self.accuracy = 0
            self.f1_score = 0
            self.xai_precision = 0
            self.fidelity = 0

    def test_accuracy(self, G_test, y_test, vectorised=False):
        if vectorised:
            X_test = self.kernel.make_vectors_vectorised(G_test)
        else:
            X_test = self.kernel.vectorise_dataset_with_keys(G_test)

        y_pred = self.clf.predict(X_test)

        self.phi_test = X_test
        self.y_test = y_test

        return accuracy_score(y_test, y_pred)

    def test_fidelity(self, baseline_acc):
        top_features = sorted(self.importances)[-self.k:]
        indices = np.where(np.isin(self.importances, top_features))[0]

        self.phi_train[:, indices] = 0

        # if self.phi_test is None:
        #     raise Error("Invalid fidelity call! Variables not set")

        clf = self.train_model(self.model, self.phi_train, self.y_train, self.best_params)
        y_pred = clf.predict(self.phi_test)
        acc = accuracy_score(self.y_test, y_pred)

        # print("\t Testing Fidelity:")
        # print(f"\t baseline acc: {baseline_acc}")
        # print(f"\t acc: {acc}")

        return baseline_acc - acc
