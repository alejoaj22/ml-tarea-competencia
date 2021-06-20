"""
In this module we store prepare the sataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    y = df["y"]
    X = df.drop(columns=["y"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def get_dataset_mod(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset_2(df)
    y = df["y"]
    X = df.drop(columns=["y"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=X["age_range"]
    )
    split_mapping = {"train": (X, y), "test": (X, y)}
    return {k: split_mapping[k] for k in splits}



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            remover_overlines,
            _columnas_a_numericas,
            _columnas_a_string,
            _fix_unhandled_nulls,
            reemplazar_vacios,
            _add_agecategorical,
            _add_chol_log,
            remover_columnas
            


        ]
    )
    df = cleaning_fn(df)
    return df


def clean_dataset_2(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
#            remover_overlines,
            _columnas_a_numericas,
            _columnas_a_string,
            _fix_unhandled_nulls,
            reemplazar_vacios,
            _add_agecategorical,
            _add_chol_log,
            remover_columnas
            


        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def reemplazar_vacios(df):
    df = df.fillna(0)
    return df

def remover_columnas(df):
    df = df.drop(["trtbps", "chol", "fbs"], axis = 1)
    return df

def remover_overlines(df):
    z = np.abs(stats.zscore(df))
    df = df[(z<3).all(axis=1)]
    return df


def _columnas_a_numericas(df):
    df[["age","oldpeak","thalachh"]] = df[["age","oldpeak","thalachh"]].astype(int)
    return df

def agecategorical(age):
    if age <45:
        return "0"
    elif age >=45:
        return "1"

def _add_agecategorical(df):
    df['age_range'] = df['age'].apply(agecategorical)
    return df

def transform_chol(chol):
    return np.log(chol)

def _add_chol_log(df):
    df['chollog'] = df['chol'].apply(transform_chol)
    return df

def _columnas_a_string(df):
    df[["cp","restecg","slp","caa","thall","sex","exng"]] = df[["cp","restecg","slp","caa","thall","sex","exng"]].astype(str)
    #df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]] = df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]]
    return df

def _fix_unhandled_nulls(df):
    df.fillna(0, inplace=True)
    return df

def get_categorical_column_names() -> t.List[str]:
    return ("age,cp,restecg,slp,caa,thall").split(",")


def get_binary_column_names() -> t.List[str]:
    return ("sex,exng").split(",")


def get_numeric_column_names() -> t.List[str]:
    return ("age,oldpeak,thalachh").split(",")


def get_column_names() -> t.List[str]:
    return ("age,sex,cp,restecg,thalachh,exng,oldpeak,slp,caa,thall").split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
#        "age": ('54', '58', '47', '45', '50', '62', '65', '43',  '39', '63', '44', '68', '67', '55',  '59', '57', '56', '70', '61', '52', '60', '74', '35', '51', '64', '42', '41', '37', '53', '49', '46', '66', '48', '77', '38', '34', '71', '69'),
        "cp": ('2', '0', '1', '3'),
#        "trtbps": ('150', '120', '112', '114', '100', '130', '124', '138', '108', '140', '118', '128', '110', '134', '155', '126', '132', '136', '160', '123', '152', '122', '192', '180', '115', '135', '142', '170', '145', '125', '148', '172', '146', '178', '156', '174', '164', '144', '101', '94'),
        "restecg": ('0', '1', '2'),
        "slp": ('2', '0', '1'),
        "caa": ('0', '3', '1', '2'),
        "thall": ('3', '2', '1', '0'),
        "sex":('0', '1'),
        "fbs":('0', '1'),
        "exng":('0', '1'),
    }
