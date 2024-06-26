import json
from fastapi.testclient import TestClient
from app import app
import pytest


@pytest.fixture(scope="session")
def client():
    """FastAPI test client."""
    return TestClient(app)


def test_index(client: TestClient) -> None:

    index_string = "Hello World"

    r = client.get("/")

    assert r.status_code == 200
    assert r.json()["message"] == index_string


def test_make_prediction_single(client: TestClient) -> None:
    example = {
        "age": 27,
        "workclass": "Private",
        "education_num": 8,
        "occupation": "Exec-managerial",
        "capital_gain": 0,
        "capital_loss": 100,
        "hours_per_week": 12,
    }

    data = json.dumps(example)
    r = client.post("/inference", content=data)

    assert r.status_code == 200
    assert r.json() is not None


def test_get_metrics(client: TestClient) -> None:

    r = client.get("/metrics")

    assert r.status_code == 200
    assert "f1" in dict(r.json()).keys()


def test_fairness(client: TestClient) -> None:
    body = {"column": "occupation"}

    occupation_options = [
        'Adm-clerical', 
        'Craft-repair', 
        'Exec-managerial', 
        'Farming-fishing', 
        'Handlers-cleaners', 
        'Other-service', 
        'Priv-house-serv', 
        'Prof-specialty', 
        'Protective-serv', 
        'Sales', 
        'Tech-support', 
        'Transport-moving'
    ]

    data = json.dumps(body)

    r = client.post("/fairness", content=data)

    assert r.status_code == 200

    response_keys = dict(r.json()).keys()

    for option in occupation_options:
        assert option in response_keys

