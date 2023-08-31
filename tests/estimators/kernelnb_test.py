import platform
import unittest
from sklearn.utils.estimator_checks import check_estimator, load_iris
from sklearn.metrics import cohen_kappa_score
import numpy as np

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

        for clf in self.get_estimators():
            clf.fit(X,y)
            y_pred = clf.predict(X)

            metric_val = cohen_kappa_score(y, y_pred)
            self.assertTrue(metric_val>0, "Classifier should be better than random!")

            probas = clf.predict_proba(X)

            self.assertIsNotNone(probas, "Probabilites are nont")
            self.assertFalse(  np.isnan( probas).any(), "NaNs in probability predictions" )
            self.assertTrue( probas.shape[0] == X.shape[0], "Different number of objects in prediction" )
            self.assertTrue( probas.shape[1] == n_classes, "Wrong number of classes in proba prediction")

            prob_sums = np.sum(probas,axis=1)
            self.assertTrue( np.allclose(prob_sums, np.asanyarray( [1 for _ in range(X.shape[0])] )), "Not all sums close to one" )
            


if __name__ == '__main__':
    unittest.main()