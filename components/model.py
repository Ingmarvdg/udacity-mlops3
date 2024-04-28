from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

import pickle
import os

from typing import Optional

def get_model_pipeline(model: BaseEstimator, param_grid: Optional[dict] = None) -> Pipeline:
    """ Generate the model pipeline to be used for training and inference.

    If param_grid is provided the model is turned into a gridsearch, during
    training it will use 5-fold cross validation to try out all parameters.

    Parameters
    ----------
    model : BaseEstimator
        The estimator to be used in the model pipeline.
    param_grid : dict, optional
        An optional parameter grid to use gridsearch with, by default None

    Returns
    -------
    Pipeline
        The model pipeline.
    """
    # Let's handle the categorical features first
    ordinal_categorical = ["workclass"]
    non_ordinal_categorical = ["occupation"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = Pipeline([
        ("encode", OneHotEncoder())
        ])

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    if param_grid:
        model = GridSearchCV(model, param_grid=param_grid, cv=5, refit=True)

    sk_pipe = Pipeline([
        ("preprocessing", preprocessor), 
        ("inference", model)])
    
    return sk_pipe

def save_model(model: BaseEstimator, path: str, name: str) -> None:
    """ Save model as pickle to given file location.

    Parameters
    ----------
    model : BaseEstimator
        The model to save.
    path : str
        The folder to save the model in.
    name : str
        The filename.
    """
    with open(os.path.join(path, name), "wb") as file:
        pickle.dump(model, file)

def load_model(path: str, name: str) -> BaseEstimator:
    """ Load model from file.

    Parameters
    ----------
    path : str
        The folder where the model is stored.
    name : str
        The filename.

    Returns
    -------
    BaseEstimator
        The model.
    """
    with open(os.path.join(path, name), "rb") as file:
        model = pickle.load(file)

    return model