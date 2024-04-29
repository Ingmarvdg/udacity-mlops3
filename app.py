from fastapi import FastAPI
from components import dataloader, model, scoring, utilities
from sklearn.ensemble import RandomForestClassifier

from contextlib import asynccontextmanager
import pandas as pd

API_PROJECT_NAME="census_dummy_model"
MODEL_FOLDER = "model"
DATA_PATH = "./data/census.csv"
TEST_DATA_PATH = "../data/test_census.csv"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["census_model"] = model.load_model(MODEL_FOLDER, "basic_model.pkl")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# FastAPI app
app = FastAPI(title=API_PROJECT_NAME, lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def inference(item: utilities.Person) -> dict[str,int]:
    df = pd.DataFrame([dict(item)])
    df.columns = df.columns.str.replace("_", "-")
    r = ml_models["census_model"].predict([dict(item)])
    return {"result": r[0]}

@app.post("/fairness")
async def fairness(column: str) -> dict[str, float]:
    """ Get scores for each value in input column.

    Parameters
    ----------
    column : str
        The name of the input column.

    Returns
    -------
    dict[str, float]
        The scores for each unique value in given input column.
    """

    model = ml_models["census_model"]

    x_test, y_test = dataloader.load_test_data(TEST_DATA_PATH)

    f1_score_slices = scoring.score_model_slices(model, x_test, y_test, column)
    
    return f1_score_slices

@app.post("/train")
async def train(train_config: utilities.TrainConfig) -> None:
    x_train, x_test, y_train, y_test = dataloader.load_trainval_data(DATA_PATH, train_config.train_split)
    # grid search for random forests
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    x_train, x_test, y_train, y_test = dataloader.load_trainval_data(DATA_PATH)

    clf = RandomForestClassifier(random_state=42)

    pipe = model.get_model_pipeline(clf, param_grid)

    pipe = pipe.fit(x_train, y_train)

    # evaluate model
    f1_score = scoring.score_model(pipe, x_test, y_test)
    print(f"Got score {f1_score}")

    # save model
    model.save_model(pipe, "./model", "basic_model.pkl")
