import pytest
from sklearn.linear_model import LinearRegression
import pandas as pd

@pytest.fixture
def trainval_data_path():
    return "data/census.csv"

@pytest.fixture
def test_data_path():
    return "data/test_census.csv"

@pytest.fixture
def test_data():
    return pd.read_csv("tests/resources/test_data.csv")

@pytest.fixture
def resources():
    return "tests/resources"

@pytest.fixture
def model_artifact():
    return "tests/resources/model.pkl"

@pytest.fixture
def dummy_model():
    return LinearRegression()