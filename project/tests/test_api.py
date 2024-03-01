import json
import pytest
from fastapi.testclient import TestClient
from app import app

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

def test_make_prediction_single(client: TestClient) -> None:
    example = {
                    "age": 27,
                    "race": "White",
                    "workclass": "Private",
                    "education_num": 8,
                    "capital_gain": 0,
                    "capital_loss": 100,
                    "hours_per_week": 12
                }

    data = json.dumps(example)
    r = client.post("/inference", content=data)

    assert r.status_code == 200
    assert r.json()["result"] is not None
