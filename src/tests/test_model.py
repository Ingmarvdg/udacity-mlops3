from ..census_model.train_model import load_data, save_model, load_model

import pandas as pd
import pytest
import os
from sklearn.linear_model import LinearRegression

@pytest.fixture
def test_data():
    return "project/data/census.csv"

@pytest.fixture
def resources():
    return "project/tests/resources"

@pytest.fixture
def model_artifact():
    return "project/tests/resources/model.pkl"

@pytest.fixture
def dummy_model():
    return LinearRegression()

def test_load_data(test_data):
    train_x, test_x, _, _ = load_data(test_data, test_size=0.4)

    assert len(train_x.columns) == 8
    assert len(train_x.columns) == len(test_x.columns)

def test_save_model(resources, dummy_model):
    new_name = "model2.pkl"

    save_model(dummy_model, resources, new_name)

    files_in_dir = os.listdir(resources)
    assert new_name in files_in_dir

    # if test fails it wont be removed but whatever I don't have time.
    os.remove(os.path.join(resources, new_name))

def test_load_model(resources):
    model = load_model(resources, "model.pkl")

    assert model is not None
