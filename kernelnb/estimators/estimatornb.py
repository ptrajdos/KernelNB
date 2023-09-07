import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin,check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.neighbors import KernelDensity

from sklearn.preprocessing import normalize
from scipy.special import softmax
from scipy._lib._util import _asarray_validated


class EstimatorNB(BaseEstimator, ClassifierMixin):
    
    def __init__(self, estimator_class=KernelDensity, estimator_parameters = {}) -> None:

        """
         Naive Bayes algorithm using some density estimator

        Parameters
        ----------
        estimator_class -- Class of the estimator that will be used
        estimator_parameters  -- Paramesters of the used estimator

    """
        super().__init__()
        self.estimator_class = estimator_class
        self.estimator_parameters = estimator_parameters


    def _calc_priors(self,y):
        self.classes_, counts = np.unique(y,return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.class_priors_ = np.log(counts/np.sum(counts))


    def _create_cond_probs_kdes(self,X,y):
        self._check_n_features(X,True)
        n_objects = X.shape[0] 
        #Array of conditional probability estimators P(X|C)
        self.estimators_ = np.empty((self.n_classes_, self.n_features_in_), dtype=object)
        for class_idx in range(self.n_classes_):
            for attrib_idx in range(self.n_features_in_):

                self.estimators_[class_idx, attrib_idx] = self.estimator_class(**self.estimator_parameters)

                row_select = y == np.asanyarray( [ self.classes_[class_idx] for _ in range(n_objects) ] ) 
                data_subset= X[row_select,attrib_idx:(attrib_idx+1)]
                self.estimators_[class_idx, attrib_idx].fit(data_subset)

    def _check_regression(self,y):
        if np.issubsctype(y.dtype, np.floating):
            y_i = y.astype('int')
            if not np.allclose(y,y_i):
                raise ValueError("Unknown label type: {}".format(y.dtype))


    def fit(self, X, y):

        X, y = self._validate_data(X,y, accept_sparse=False,
                                    order="C", accept_large_sparse=False,y_numeric=False)
        
        self._check_regression(y)
        
        self._calc_priors(y)
        self._create_cond_probs_kdes(X,y)
       
        return self

    def _softmax(self,x,axis=None):
        x = _asarray_validated(x, check_finite=False)
        
        min_val = np.finfo(x.dtype).min
        x[x == -np.inf] = min_val
        x_max = np.amax(x, axis=axis, keepdims=True)
        
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
        

    def _predict_proba(self, X):

        n_objects = _num_samples(X)

        cond_probs_logs = np.zeros((n_objects,self.n_classes_, self.n_features_in_))

        for class_idx in range(self.n_classes_):
            for attrib_idx in range(self.n_features_in_):
                cond_probs_logs[:,class_idx,attrib_idx] = self.estimators_[class_idx,attrib_idx].score_samples(X[:,attrib_idx:(attrib_idx+1)])

    
        acc_probs_logs = np.sum(cond_probs_logs, axis=2)
        acc_probs_logs[:,::] += self.class_priors_
        post_probs = self._softmax(acc_probs_logs,axis=1)

        return post_probs

    def _validate_predict_input(self,X):
        
        X= self._validate_data(X, accept_sparse=False,
                                    order="C", accept_large_sparse=False,reset=False)
        
        
        check_is_fitted(self, ("classes_", "n_classes_","class_priors_", "n_features_in_","estimators_"))

        self._check_n_features(X,False)
        return X

    def predict_proba(self,X):
        self._validate_predict_input(X)
        return self._predict_proba(X)

    def predict(self,X):
        
        X = self._validate_predict_input(X)

        post_probs = self._predict_proba(X)
        class_indices = np.argmax(post_probs,axis=1)
        response = np.asanyarray( [self.classes_[class_idx] for class_idx in class_indices ])
        
        return response
    
    def _more_tags(self):
        return {
            "_xfail_checks":{
                "check_parameters_default_constructible":
                    "transformer has 1 mandatory parameter",
            }
        }
    