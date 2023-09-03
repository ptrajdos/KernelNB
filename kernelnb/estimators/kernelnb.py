import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin,check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.neighbors import KernelDensity

from sklearn.preprocessing import normalize
from scipy.special import softmax


class KernelNB(BaseEstimator, ClassifierMixin):
    
    def __init__(self, bandwidth="silverman", algorithm="auto",kernel="gaussian", metric="euclidean",
                 atol=0, rtol=0, breadth_first=True, leaf_size=40,metric_params=None) -> None:

        """
         Naive Bayes algorithm using Kernel Density Estimators

        Parameters
        ----------
        bandwidth : float or {"scott", "silverman"}, default="silverman"
            The bandwidth of the kernel. If bandwidth is a float, it defines the
            bandwidth of the kernel. If bandwidth is a string, one of the estimation
            methods is implemented.

        algorithm : {'kd_tree', 'ball_tree', 'auto'}, default='auto'
            The tree algorithm to use.

        kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
                    'cosine'}, default='gaussian'
            The kernel to use.

        metric : str, default='euclidean'
            Metric to use for distance computation. See the
            documentation of `scipy.spatial.distance
            <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
            the metrics listed in
            :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
            values.

            Not all metrics are valid with all algorithms: refer to the
            documentation of :class:`BallTree` and :class:`KDTree`. Note that the
            normalization of the density output is correct only for the Euclidean
            distance metric.

        atol : float, default=0
            The desired absolute tolerance of the result.  A larger tolerance will
            generally lead to faster execution.

        rtol : float, default=0
            The desired relative tolerance of the result.  A larger tolerance will
            generally lead to faster execution.

        breadth_first : bool, default=True
            If true (default), use a breadth-first approach to the problem.
            Otherwise use a depth-first approach.

        leaf_size : int, default=40
            Specify the leaf size of the underlying tree.  See :class:`BallTree`
            or :class:`KDTree` for details.

        metric_params : dict, default=None
            Additional parameters to be passed to the tree for use with the
            metric.  For more information, see the documentation of
            :class:`BallTree` or :class:`KDTree`.

    """
        super().__init__()
        self.bandwidth=bandwidth
        self.algorithm=algorithm
        self.kernel=kernel
        self.metric=metric
        self.atol=atol
        self.rtol=rtol
        self.breadth_first=breadth_first
        self.leaf_size=leaf_size
        self.metric_params=metric_params


    def _calc_priors(self,y):
        self.classes_, counts = np.unique(y,return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.class_priors_ = counts/np.sum(counts)


    def _create_cond_probs_kdes(self,X,y):
        self._check_n_features(X,True)
        n_objects = X.shape[0] 
        #Array of conditional probability estimators P(X|C)
        self.kernel_estimators_ = np.empty((self.n_classes_, self.n_features_in_), dtype=object)
        for class_idx in range(self.n_classes_):
            for attrib_idx in range(self.n_features_in_):

                self.kernel_estimators_[class_idx, attrib_idx] = KernelDensity(
                    bandwidth=self.bandwidth, 
                    algorithm=self.algorithm,
                    kernel=self.kernel,
                    metric=self.metric,
                    atol=self.atol,
                    rtol=self.rtol,
                    breadth_first=self.breadth_first,
                    leaf_size=self.leaf_size,
                    metric_params=self.metric_params
                )
                row_select = y == np.asanyarray( [ self.classes_[class_idx] for _ in range(n_objects) ] ) 
                data_subset= X[row_select,attrib_idx:(attrib_idx+1)]
                self.kernel_estimators_[class_idx, attrib_idx].fit(data_subset)

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

    def _predict_proba(self, X):

        n_objects = _num_samples(X)

        cond_probs_logs = np.zeros((n_objects,self.n_classes_, self.n_features_in_))

        for class_idx in range(self.n_classes_):
            for attrib_idx in range(self.n_features_in_):
                cond_probs_logs[:,class_idx,attrib_idx] = self.kernel_estimators_[class_idx,attrib_idx].score_samples(X[:,attrib_idx:(attrib_idx+1)])

        acc_probs_logs = np.sum(cond_probs_logs, axis=2)
        acc_probs_logs[:,::] += self.class_priors_
        post_probs = softmax(acc_probs_logs,axis=1)

        return post_probs

    def _validate_predict_input(self,X):
        
        X= self._validate_data(X, accept_sparse=False,
                                    order="C", accept_large_sparse=False,reset=False)
        
        
        check_is_fitted(self, ("classes_", "n_classes_","class_priors_", "n_features_in_","kernel_estimators_"))

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
    