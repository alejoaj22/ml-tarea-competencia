"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LinearRegression,SGDClassifier,RidgeClassifier,LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import data


EstimatorConfig = t.List[t.Dict[str, t.Any]]


def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        params = step["params"]
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "random-forest-regressor": RandomForestRegressor,
        "linear-regressor": LinearRegression,
        "average-price-per-neighborhood-regressor": AveragePricePerNeighborhoodRegressor,
        "categorical-encoder": CategoricalEncoder,
        "standard-scaler": StandardScaler,
        "discretizer": _get_discretizer,
        "crosser": _get_crosser,
        "PolynomialFeatures": PolynomialFeatures,
        "averager": AveragePricePerNeighborhoodExtractor,
        "decision-treee-regresor" : DecisionTreeRegressor,
        "lineal-classifier" : RidgeClassifier,
        "SGDClassifier" : SGDClassifier,
        "Random-forest-classifier": RandomForestClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "Logis-ticRegression": LogisticRegression,
        "base-clasificador": modelobasedeclasificacion
    }


def _get_discretizer(
    *,
    bins_per_column: t.Mapping[str, int],
    encode: str = "onehot",
    stratey: str = "quantile",
):
    columns, n_bins = zip(*bins_per_column.items())
    transformer = ColumnTransformer(
        [
            (
                "discretizer",
                KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=stratey),
                columns,
            )
        ],
        remainder="drop",
    )

    return transformer


def _get_crosser(
    *,
    columns: t.Sequence[int],
):
    transformer = ColumnTransformer(
        ("crosser", PolynomialFeatures(interaction_only=True), columns),
        remainder="passthrough",
    )
    return transformer


class CreateNewColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["agebin"] = pd.cut(X['age'].astype(int), bins=[0,20,45,100], labels=["Low", "Mid", "High"])
        X["cholesterollog"] = (X['chol'].astype(int)+1).transform(np.log)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        one_hot: bool = False,
        force_dense_array: bool = False,
        pass_through_columns: t.Optional[t.Sequence[str]] = None,
    ):
        self.one_hot = one_hot
        self.force_dense_array = force_dense_array
        self.pass_through_columns = pass_through_columns
        self.categorical_column_names = (
            data.get_binary_column_names() + data.get_categorical_column_names()
        )
        mapping = data.get_categorical_variables_values_mapping()
        self.categories = [mapping[k] for k in self.categorical_column_names]

    def fit(self, X, y=None):
        X = X.copy()
        self.n_features_in_ = X.shape[1]
        pass_through_columns = data.get_numeric_column_names()
        if self.pass_through_columns is not None:
            pass_through_columns = pass_through_columns + self.pass_through_columns
        encoder_cls = (
            partial(OneHotEncoder, drop="first", sparse=not self.force_dense_array)
            if self.one_hot
            else OrdinalEncoder
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    encoder_cls(
                        categories=self.categories,
                    ),
                    self.categorical_column_names,
                ),
                ("pass-numeric", "passthrough", pass_through_columns),
            ],
            remainder="drop",
        )
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)


class AveragePricePerNeighborhoodRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        """Computes the mode of the price per neighbor on training data."""
        df = pd.DataFrame({"Neighborhood": X["Neighborhood"], "price": y})
        self.means_ = df.groupby("Neighborhood").mean().to_dict()["price"]
        self.global_mean_ = y.mean()
        return self

    def predict(self, X):
        """Predicts the mode computed in the fit method."""

        def get_average(x):
            if x in self.means_:
                return self.means_[x]
            else:
                return self.global_mean_

        y_pred = X["Neighborhood"].apply(get_average)
        return y_pred



class modelobasedeclasificacion(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        """Computes the mode of the price per neighbor on training data."""
        df = pd.DataFrame({"age": X["age"],"sex" : X["sex"],"cp" : X["cp"],"restecg" : X["restecg"],"thalachh" : X["thalachh"],"oldpeak" : X["oldpeak"],"slp" : X["slp"],"caa" : X["caa"],"thall" : X["thall"], "y": y})
        self.means_ = df.groupby("y").mean().to_dict()
        self.global_mean_ = y.mean()

        return self

    def predict(self, X):
        """Predicts the mode computed in the fit method."""
        def transform(self, X):
            y = []
            aux = np.abs(X["age"] - self.means_.get("age")[0])
            aux_2 = np.abs(X["age"] - self.means_.get("age")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)
            
            aux = np.abs(X["sex"] - self.means_.get("sex")[0])
            aux_2 = np.abs(X["sex"] - self.means_.get("sex")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)
            
            aux = np.abs(X["cp"] - self.means_.get("cp")[0])
            aux_2 = np.abs(X["cp"] - self.means_.get("cp")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)

            aux = np.abs(X["restecg"] - self.means_.get("restecg")[0])
            aux_2 = np.abs(X["restecg"] - self.means_.get("restecg")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)
            
            aux = np.abs(X["thalachh"] - self.means_.get("thalachh")[0])
            aux_2 = np.abs(X["thalachh"] - self.means_.get("thalachh")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)

            aux = np.abs(X["oldpeak"] - self.means_.get("oldpeak")[0])
            aux_2 = np.abs(X["oldpeak"] - self.means_.get("oldpeak")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)
            

            aux = np.abs(X["slp"] - self.means_.get("slp")[0])
            aux_2 = np.abs(X["slp"] - self.means_.get("slp")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)

            aux = np.abs(X["caa"] - self.means_.get("caa")[0])
            aux_2 = np.abs(X["caa"] - self.means_.get("caa")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)
            
            aux = np.abs(X["thall"] - self.means_.get("thall")[0])
            aux_2 = np.abs(X["thall"] - self.means_.get("thall")[1])
            if aux < aux_2:
                y.append(1)
            else:
                y.append(0)

            y = np.array(y)

            y_counta_zeros = np.count_nonzero(y == 0)
            y_counta_unos = np.count_nonzero(y == 1)


            if y_counta_zeros < y_counta_unos:
                y_pred = 1
            else:
                y_pred = 0 
            return  y_pred

        y_pred = X.apply(transform)
        return y_pred

class AveragePricePerNeighborhoodExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        df = pd.DataFrame({"Neighborhood": X["Neighborhood"], "price": y})
        self.means_ = df.groupby("Neighborhood").mean().to_dict()["price"]
        self.global_mean_ = y.mean()
        return self

    def transform(self, X):
        def get_average(x):
            if x in self.means_:
                return self.means_[x]
            else:
                return self.global_mean_

        X["AveragePriceInNeihborhood"] = X["Neighborhood"].apply(get_average)
        return X
