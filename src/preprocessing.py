from typing import List

import pandas as pd


def drop_50_perc_null_features(df: pd.DataFrame, perc: float = 0.5) -> int:
    """Drop features with more than 50% null values.

    :param df: The dataframe from which to drop features.
    :param perc: The minimum percentage of null values for the feature.
    :return: Number of features dropped.
    """
    len_before = df.columns.size
    threshold = len(df) * perc
    df.dropna(thresh=threshold, axis=1, inplace=True)
    return len_before - df.columns.size


def drop_features_by_keywords(
    df: pd.DataFrame, keywords: List[str], show: bool = False
) -> int:
    """Drop features that contain certain keywords in their name.

    :param df: The dataframe from which to drop features.
    :param keywords: The list of keywords to search for in feature names.
    :param show: True if you want to show the dropped features names.
    :return: Number of features dropped.
    """
    cols_to_drop = [
        col for col in df.columns if any(key in col.lower() for key in keywords)
    ]

    if show:
        print("Columns removed:")
        print(cols_to_drop)

    df.drop(columns=cols_to_drop, inplace=True)
    return len(cols_to_drop)


if __name__ == "__main__":
    df = pd.read_csv("data/raw.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nDropping columns that would cause look-ahead bias...")
    removed = drop_features_by_keywords(
        df, keywords=["r1", "r2", "r3", "r4", "r5"], show=False
    )
    print(f"{removed} rounds-related features removed!")
    removed = drop_features_by_keywords(df, keywords=["odds"], show=False)
    print(f"{removed} odds-related features removed!")

    print("\nDropping columns with more than 50% null values...")
    removed = drop_50_perc_null_features(df)
    print(f"{removed} columns removed!")
    print(f"Dataset shape: {df.shape}")

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
