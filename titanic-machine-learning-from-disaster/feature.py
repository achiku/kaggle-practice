import pandas as pd


def calc_family_size(x):
    return x['Sibsp'] + x['Prch'] + 1


def is_alone(x):
    if x['Sibsp'] == 0 and x['Prch'] == 0:
        return 1
    return 0


def fill_nan_fare(df_org, fill_fare):
    df = df_org.copy()
    return df['Fare'].fillna(fill_fare)


def fill_nan_embarked(df_org, fill_embarked):
    df = df_org.copy()
    return df['Embarked'].fillna(fill_embarked)


def get_categorical_fare(df_org):
    df = df_org.copy()
    return pd.qcut(df['Fare'], 4)
