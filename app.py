from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
from project.census_model.train_model import load_model

class Person(BaseModel):
    age: int
    race: str
    workclass: str
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 27,
                    "race": "White",
                    "workclass": "Private",
                    "education_num": 8,
                    "capital_gain": 0,
                    "capital_loss": 100,
                    "hours_per_week": 12
                }
            ]
        }
    }

API_PROJECT_NAME="census_dummy_model"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["census_model"] = load_model("project/model", "basic_model.pkl")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# FastAPI app
app = FastAPI(title=API_PROJECT_NAME, lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def inference(item: Person) -> dict[str,int]:
    df = pd.DataFrame([dict(item)])
    df.columns = df.columns.str.replace("_", "-")
    r = ml_models["census_model"].predict(df)
    return {"result": r[0]}