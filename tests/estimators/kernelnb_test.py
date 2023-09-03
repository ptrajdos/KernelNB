import platform
import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import cohen_kappa_score
from sklearn.datasets import load_iris, load_digits
import numpy as np
from sklearn.model_selection import train_test_split

from kernelnb.estimators.kernelnb import KernelNB

class KernelNBTest(unittest.TestCase):

    def get_estimators(self):

        return[
            KernelNB()
        ]
    

    def test_sklearn(self):

        #TODO Need to be fixed in more elegant way
        if platform.system() == 'FreeBSD':
            raise unittest.SkipTest("Skipping test according to a bug in threadpoolctl! ")

        for clf in self.get_estimators():
            for estimator, check in check_estimator(clf, generate_only=True):
                check(estimator)

    def test_iris(self):
        X, y = load_iris(return_X_y=True)
        n_classes = len( np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)

        for clf in self.get_estimators():
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            self.assertTrue(metric_val>0, "Classifier should be better than random!")

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(  np.isnan( probas).any(), "NaNs in probability predictions" )
            self.assertTrue( probas.shape[0] == X.shape[0], "Different number of objects in prediction" )
            self.assertTrue( probas.shape[1] == n_classes, "Wrong number of classes in proba prediction")

            prob_sums = np.sum(probas,axis=1)
            self.assertTrue( np.allclose(prob_sums, np.asanyarray( [1 for _ in range(X.shape[0])] )), "Not all sums close to one" )
            

    def test_digits(self):
        X, y = load_digits(return_X_y=True)
        n_classes = len( np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)

        for clf in self.get_estimators():
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)

            metric_val = cohen_kappa_score(y_test, y_pred)
            self.assertTrue(metric_val>0, "Classifier should be better than random!")

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, "Probabilites are None")
            self.assertFalse(  np.isnan( probas).any(), "NaNs in probability predictions" )
            self.assertTrue( probas.shape[0] == X.shape[0], "Different number of objects in prediction" )
            self.assertTrue( probas.shape[1] == n_classes, "Wrong number of classes in proba prediction")

            prob_sums = np.sum(probas,axis=1)
            self.assertTrue( np.allclose(prob_sums, np.asanyarray( [1 for _ in range(X.shape[0])] )), "Not all sums close to one" )


if __name__ == '__main__':
    unittest.main()