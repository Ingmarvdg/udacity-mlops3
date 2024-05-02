from components import dataloader
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = 8


def test_preprocessing(test_data: pd.DataFrame):
    x, y = dataloader.preprocess_input_data(test_data)

    assert len(x.columns) == EXPECTED_COLUMNS
    assert len(x) == len(y)


def test_trainval_data(trainval_data_path: str):
    test_size = 0.3
    train_x, test_x, train_y, test_y = dataloader.load_trainval_data(
        trainval_data_path, test_size
    )

    assert len(train_x.columns) == EXPECTED_COLUMNS
    assert len(test_x.columns) == EXPECTED_COLUMNS
    assert np.isclose(len(test_y) / (len(train_y) + len(test_y)), test_size, atol=0.02)  # noqa: E501


def test_test_data(test_data_path: str):
    x, y = dataloader.load_test_data(test_data_path)

    assert len(x.columns) == EXPECTED_COLUMNS
    assert len(x) == len(y)
