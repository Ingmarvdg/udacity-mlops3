import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pickle

# load data
def load_data(data_path, test_size=0.2):
    df = pd.read_csv(data_path)
    df["50kplus"] = df["salary"] == ">50K"
    df["50kplus"] = df["50kplus"].astype(int)

    x = df.select_dtypes(include='number')

    y = x.pop("50kplus")

    return train_test_split(x, y, test_size=0.2)

def get_trained_model(x_train, y_train):
    model = LogisticRegression()

    model.fit(x_train, y_train)

    return model

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

    f1_score = metrics.f1_score(predictions, y_test)
    
    return f1_score

def score_model_slices(model, x_test, y_test) -> dict:
    scores = {}
    for c in [0, 1]:
        mask = y_test==c
        x = x_test[mask]
        y = y_test[mask]

        scores[c] = score_model(model, x, y)

    return scores

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data("../data/census.csv")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = get_trained_model(x_train, y_train)

    # evaluate model
    f1_score = score_model(model, x_test, y_test)
    print(f"Got score {f1_score}")

    f1_score_slices = score_model_slices(model, x_test, y_test)
    print(f"Got scores {f1_score_slices} for both slices.")
    # save model
    save_model(model, "../model", "basic_model.pkl")