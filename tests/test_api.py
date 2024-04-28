import json
from fastapi.testclient import TestClient
from app import app
from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def client():
    """FastAPI test client."""
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}


def test_index(client: TestClient) -> None:

    index_string = "Hello World"

    r = client.get("/")

    assert r.status_code == 200
    assert r.json()["message"] == index_string

def test_train_model(client: TestClient) -> None:
    body = {
        "train_split": 0.2
    }

    data = json.dumps(body)
    r = client.post("/train", content=data, timeout=5)

    assert r.status_code == 200
    assert r.json()["result"] is not None


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
    assert r.json()["result"] is not None

# def test_train(client: TestClient) -> None:
#     ...