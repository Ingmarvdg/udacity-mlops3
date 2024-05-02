import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional

USEFUL_COLUMNS = [
    "age",
    "workclass",
    "education-num",
    "occupation",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "salary",
]


def preprocess_input_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess input data.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input dataframe

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The preprocessed data and respective labels.
    """

    df = df[USEFUL_COLUMNS]

    df["50kplus"] = df["salary"] == ">50K"
    df["50kplus"] = df["50kplus"].astype(int)

    y = df.pop("50kplus")

    return df, y


def load_test_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the test data from path.

    Parameters
    ----------
    data_path : str
        Path of the test data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The preprocessed test data and respective labels.
    """
    df = pd.read_csv(data_path)

    x, y = preprocess_input_data(df)

    return x, y


def load_trainval_data(
    data_path: str, test_size: Optional[float] = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the trainval data from the path in the repository.

    Parameters
    ----------
    data_path : str
        The path to the data in the repository.
    test_size : Optional[float], optional
        The size of the test set, by default 0.2

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the x_train, y_train, x_test, and y_test data.
    """

    df = pd.read_csv(data_path)

    x, y = preprocess_input_data(df)

    return train_test_split(x, y, test_size=test_size)
