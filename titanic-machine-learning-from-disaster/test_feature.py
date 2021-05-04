import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
import pytest


@pytest.mark.parametrize(
    "x, expected",
    [
        (pd.Series({'Sibsp': 1, 'Prch': 1}, dtype='int64'), np.int64(3)),
        (pd.Series({'Sibsp': 1, 'Prch': 0}, dtype='int64'), np.int64(2)),
        (pd.Series({'Sibsp': 0, 'Prch': 1}, dtype='int64'), np.int64(2)),
        (pd.Series({'Sibsp': 0, 'Prch': 0}, dtype='int64'), np.int64(1)),
    ],
)
def test_calc_family_size(x, expected):
    from feature import calc_family_size
    assert calc_family_size(x) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (pd.Series({'Sibsp': 0, 'Prch': 0}, dtype='int64'), 1),
        (pd.Series({'Sibsp': 1, 'Prch': 1}, dtype='int64'), 0),
        (pd.Series({'Sibsp': 0, 'Prch': 1}, dtype='int64'), 0),
        (pd.Series({'Sibsp': 1, 'Prch': 0}, dtype='int64'), 0),
    ],
)
def test_is_alone(x, expected):
    from feature import is_alone
    assert is_alone(x) == expected


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame({'Fare': [10, 20, 30, 10, np.nan]}),
            pd.Series([10, 20, 30, 10, 15], name='Fare', dtype='float64')
        ),
    ],
)
def test_fill_nan_fare(df, expected):
    from feature import fill_nan_fare
    result = fill_nan_fare(df, df['Fare'].median())
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame({'Embarked': ['Q', 'C', 'C', np.nan, np.nan]}),
            pd.Series(['Q', 'C', 'C', 'S', 'S'], name='Embarked', dtype='str')
        ),
    ],
)
def test_fill_nan_embarked(df, expected):
    from feature import fill_nan_embarked
    result = fill_nan_embarked(df, 'S')
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame({'Fare': [10, 20, 30, 40, 50, 60, 70, 80]}),
            pd.Series([0, 0, 1, 1, 2, 2, 3, 3], name='Fare'),
        ),
    ],
)
def test_get_categorical_fare(df, expected):
    from feature import get_categorical_fare
    result = get_categorical_fare(df)
    assert_series_equal(result, expected)
