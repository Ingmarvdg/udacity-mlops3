from fastapi import FastAPI
from components import dataloader, model, scoring, utilities

import pandas as pd

API_PROJECT_NAME = "census_dummy_model"
MODEL_FOLDER = "model"
DATA_PATH = "./data/census.csv"
TEST_DATA_PATH = "../data/test_census.csv"

# FastAPI app
app = FastAPI(title=API_PROJECT_NAME)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
async def inference(item: utilities.Person) -> dict[str, int]:
    df = pd.DataFrame([dict(item)])
    df.columns = df.columns.str.replace("_", "-")

    clf = model.load_model(MODEL_FOLDER, "basic_model.pkl")

    r = clf.predict(df)
    return {"result": r[0]}


@app.post("/fairness")
async def fairness(item: utilities.FairnessConfig) -> dict[str, float]:
    """Get scores for each value in input column.

    Parameters
    ----------
    column : str
        The name of the input column.

    Returns
    -------
    dict[str, float]
        The scores for each unique value in given input column.
    """

    clf = model.load_model(MODEL_FOLDER, "basic_model.pkl")

    x_test, y_test = dataloader.load_test_data(TEST_DATA_PATH)

    f1_score_slices = scoring.score_model_slices(clf,
                                                 x_test,
                                                 y_test,
                                                 item.column)

    return f1_score_slices


@app.get("/metrics")
async def train() -> dict[str, float]:
    x_test, y_test = dataloader.load_test_data(TEST_DATA_PATH)

    pipe = model.load_model("./model", "basic_model.pkl")

    # evaluate model
    f1_score = scoring.score_model(pipe, x_test, y_test)
    print(f"Got score {f1_score}")

    return {"f1": f1_score}
