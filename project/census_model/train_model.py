import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

# load data
def load_data(data_path, test_size=0.2):
    useful_columns = ["age", "race","workclass", "education-num", "capital-gain","capital-loss","hours-per-week", "salary"]

    df = pd.read_csv(data_path)

    df = df[useful_columns]

    df["50kplus"] = df["salary"] == ">50K"
    df["50kplus"] = df["50kplus"].astype(int)

    y = df.pop("50kplus")

    return train_test_split(df, y, test_size=test_size)

def get_inference_pipeline():
    # Let's handle the categorical features first
    ordinal_categorical = ["workclass"]
    non_ordinal_categorical = ["race"]
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
    model = LogisticRegression()

    sk_pipe = Pipeline([
        ("preprocessing", preprocessor), 
        ("inference", model)])

    return sk_pipe

def save_model(model, path, name):
    with open(os.path.join(path, name), "wb") as file:
        pickle.dump(model, file)

def load_model(path, name):
    with open(os.path.join(path, name), "rb") as file:
        model = pickle.load(file)

    return model

def model_inference(model, sample):
    return model.predict(sample)

def score_model(model, x_test, y_test):
    predictions = model.predict(x_test)

    actual = y_test.values

    f1_score = metrics.f1_score(predictions, actual)
    
    return f1_score

def score_model_slices(model, x_test, y_test, slice_column) -> dict:
    # get predictions and add to column
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=["predictions"])

    full_df = pd.concat([x_test.reset_index(drop=True), 
                         y_test.reset_index(drop=True), 
                         y_pred.reset_index(drop=True)
                         ], axis=1)
    

    agg_df = full_df.groupby(slice_column).apply(lambda x: metrics.f1_score(x["predictions"], x["50kplus"]))

    return agg_df

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data("../data/census.csv")

    print(x_train.columns)
    pipe = get_inference_pipeline()

    pipe = pipe.fit(x_train, y_train)

    # evaluate model
    f1_score = score_model(pipe, x_test, y_test)
    print(f"Got score {f1_score}")

    f1_score_slices = score_model_slices(pipe, x_test, y_test, "race")
    print(f"Got scores {f1_score_slices} for both slices.")
    # save model
    save_model(pipe, "../model", "basic_model.pkl")