import unittest
from sklearn.neighbors import KernelDensity
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import cohen_kappa_score
from sklearn.datasets import load_iris, load_digits
import numpy as np
from sklearn.model_selection import train_test_split
from kernelnb.estimators.estimatornb import EstimatorNB
from sklearn.mixture import GaussianMixture


class EstimatorNBTest(unittest.TestCase):

    def get_estimators(self):

        return [
            EstimatorNB(),
            EstimatorNB(
                estimator_class=KernelDensity,
                estimator_parameters={
                    "bandwidth": "silverman",
                    "kernel": "gaussian",
                    "metric": "euclidean",
                },
            ),
            EstimatorNB(
                estimator_class=GaussianMixture,
                estimator_parameters={"n_components": 1},
            ),
        ]

    def test_sklearn(self):

        for clf in self.get_estimators():
            check_estimator(clf)

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

    def test_sample(self):
        X, y = load_iris(return_X_y=True)
        
        samples_list = [1,3,10,200]

        for clf in self.get_estimators():
            with self.subTest(clf=clf):
                clf.fit(X, y)
                for n_samples in samples_list:
                    try:
                        X_s, y_s = clf.sample(n_samples=n_samples)
                        self.assertIsNotNone(X_s, "Sampled features are None")
                        self.assertIsNotNone(y_s, "Sampled labels are None")
                        self.assertFalse(np.isnan(X_s).any(), "NaNs in sampled features")
                        self.assertFalse(np.isinf(X_s).any(), "Inf in sampled features")
                        self.assertFalse(np.isnan(y_s).any(), "NaNs in sampled labels")
                        self.assertFalse(np.isinf(y_s).any(), "Inf in sampled labels")
                        self.assertEqual(X_s.shape[0], n_samples, "Wrong number of samples")
                        self.assertEqual(X_s.shape[1], X.shape[1], "Wrong number of features")
                        self.assertEqual(len(y_s), n_samples, "Wrong number of labels")
                    except NotImplementedError:
                        self.skipTest("Sample method not implemented for this estimator")

if __name__ == "__main__":
    unittest.main()
