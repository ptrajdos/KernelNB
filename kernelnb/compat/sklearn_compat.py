from packaging import version
import sklearn

sklearn_version_parsee = version.parse(sklearn.__version__)

if sklearn_version_parsee >= version.parse("1.6"):
    from sklearn.utils.validation import _check_n_features
    def check_n_features_internal(estimator,X, reset):
        _check_n_features(estimator=estimator,X=X,reset=reset)
else:
    def check_n_features_internal(estimator,X, reset):
        estimator._check_n_features(X, reset)

if sklearn_version_parsee >= version.parse("1.6"):
    from sklearn.utils.validation import validate_data
    def validate_fit_external(estimator,X, y):
            X, y = validate_data(
            estimator,
            X=X,
            y=y,
            accept_sparse=False,
            order="C",
            accept_large_sparse=False,
            y_numeric=False,
            reset=True,
        )
            return X, y
else:
        def validate_fit_external(estimator,X, y):
            X, y = estimator._validate_data(
                X,
                y,
                accept_sparse=False,
                order="C",
                accept_large_sparse=False,
                y_numeric=False,
                reset=True,
            )
            return X, y
        

if sklearn_version_parsee >= version.parse("1.6"):
    from sklearn.utils.validation import validate_data
    def validate_predict_input_ext(estimator,X):

        X = validate_data(
            estimator,
            X=X,
            accept_sparse=False,
            order="C",
            accept_large_sparse=False,
            reset=False,
        )
        return X
else:
    def validate_predict_input_ext(estimator,X):
        X = estimator._validate_data(
            X, accept_sparse=False, order="C", accept_large_sparse=False, reset=False
        )
        return X