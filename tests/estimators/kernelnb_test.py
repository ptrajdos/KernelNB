import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import cohen_kappa_score
from sklearn.datasets import load_iris, load_digits
import numpy as np
from sklearn.model_selection import train_test_split

from kernelnb.estimators.kernelnb import KernelNB


class KernelNBTest(unittest.TestCase):

    def get_estimators(self):

        return [
            KernelNB(),
            KernelNB(kernel="epanechnikov"),
            KernelNB(bandwidth="scott"),
            KernelNB(kernel="epanechnikov", bandwidth="scott"),
        ]

    def test_sklearn(self):

        for clf in self.get_estimators():
            check_estimator(clf)

    def test_iris(self):
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf in self.get_estimators():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            self.assertTrue(metric_val > 0, "Classifier should be better than random!")

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
            self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
            self.assertTrue(
                probas.shape[0] == X.shape[0],
                "Different number of objects in prediction",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                "Wrong number of classes in proba prediction",
            )
            self.assertTrue(np.all(probas >= 0), "Negative probabilities")
            self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")
            
            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                "Not all sums close to one",
            )

    def test_iris_weights(self):
        X, y = load_iris(return_X_y=True)
        n_classes = len(np.unique(y))
        n_attribs = X.shape[1]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf in self.get_estimators():
            clf.fit(X_train, y_train)
           
            random_weights = np.random.random(n_attribs)
            probas = clf._predict_proba(X, weights=random_weights)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
            self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
            self.assertTrue(
                probas.shape[0] == X.shape[0],
                "Different number of objects in prediction",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                "Wrong number of classes in proba prediction",
            )

            self.assertTrue(np.all(probas >= 0), "Negative probabilities")
            self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                "Not all sums close to one",
            )


    def test_digits(self):
        X, y = load_digits(return_X_y=True)
        n_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf in self.get_estimators():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            self.assertTrue(metric_val > 0, "Classifier should be better than random!")

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
            self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
            self.assertTrue(
                probas.shape[0] == X.shape[0],
                "Different number of objects in prediction",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                "Wrong number of classes in proba prediction",
            )
            self.assertTrue(np.all(probas >= 0), "Negative probabilities")
            self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X.shape[0])])),
                "Not all sums close to one",
            )

    def test_one_val(self):
        R = 200

        X = np.zeros((R, 2))
        X[: R // 2, 0] = 1
        X[R // 2 :, 1] = 1

        y = [1 if n >= R // 2 else 0 for n in range(R)]
        n_classes = 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        for clf in self.get_estimators():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            self.assertTrue(metric_val > 0, "Classifier should be better than random!")

            probas = clf.predict_proba(X_test)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
            self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
            self.assertTrue(
                probas.shape[0] == X_test.shape[0],
                "Different number of objects in prediction",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                "Wrong number of classes in proba prediction",
            )
            self.assertTrue(np.all(probas >= 0), "Negative probabilities")
            self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(
                    prob_sums, np.asanyarray([1 for _ in range(X_test.shape[0])])
                ),
                "Not all sums close to one",
            )

            X_n = np.random.random((100, 2)) * 2 - 1

            probas = clf.predict_proba(X_n)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(np.isnan(probas).any(), "NaNs in probability predictions")
            self.assertFalse(np.isinf(probas).any(), "Inf in probability predictions")
            self.assertTrue(
                probas.shape[0] == X_n.shape[0],
                "Different number of objects in prediction",
            )
            self.assertTrue(
                probas.shape[1] == n_classes,
                "Wrong number of classes in proba prediction",
            )

            self.assertTrue(np.all(probas >= 0), "Negative probabilities")
            self.assertTrue(np.all(probas <= 1), "Probabilities bigger than one")

            prob_sums = np.sum(probas, axis=1)
            self.assertTrue(
                np.allclose(prob_sums, np.asanyarray([1 for _ in range(X_n.shape[0])])),
                "Not all sums close to one",
            )


if __name__ == "__main__":
    unittest.main()
