from sklearn import metrics
from sklearn.base import BaseEstimator
import pandas as pd

def score_model(model: BaseEstimator, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """Return the performance of a model on the provided test dataset.

    Currently performance is only based on the F1-score.

    Parameters
    ----------
    model : BaseEstimator
        The model.
    x_test : pd.DataFrame
        The test data.
    y_test : pd.DataFrame
        The expected outputs.

    Returns
    -------
    float
        The model performance.
    """

    predictions = model.predict(x_test)

    actual = y_test.values

    f1_score = metrics.f1_score(predictions, actual)
    
    return f1_score

def score_model_slices(model: BaseEstimator, x_test: pd.DataFrame, y_test: pd.DataFrame, slice_column: str) -> dict:
    """ Get the score for each value "slice" in a given column.

    Parameters
    ----------
    model : BaseEstimator
        The model.
    x_test : pd.DataFrame
        The test data.
    y_test : pd.DataFrame
        The expected outputs.
    slice_column : str
        The column to get scores for.

    Returns
    -------
    dict
        The keys are each unique value in the column and values are the scores.
    """
    # get predictions and add to column
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=["predictions"])

    full_df = pd.concat([x_test.reset_index(drop=True), 
                         y_test.reset_index(drop=True), 
                         y_pred.reset_index(drop=True)
                         ], axis=1)
    

    agg_df = full_df.groupby(slice_column).apply(lambda x: metrics.f1_score(x["predictions"], x["50kplus"])).asdict()

    return agg_df