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
    df_raw = pd.read_csv("data/raw.csv")
    print(f"Dataset shape: {df_raw.shape}")
    print(f"Memory Usage: {df_raw.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    print("\nInformations about columns:")
    print(df_raw.info())

    print(f"\nColumns list: {list(df_raw.columns)}")

    print("\nDropping columns that would cause look-ahead bias...")
    removed = drop_features_by_keywords(
        df_raw, keywords=["r1", "r2", "r3", "r4", "r5"], show=False
    )
    print(f"{removed} rounds-related features removed!")
    removed = drop_features_by_keywords(df_raw, keywords=["odds"], show=False)
    print(f"{removed} odds-related features removed!")

    print("\nDropping columns with more than 50% null values...")
    removed = drop_50_perc_null_features(df_raw)
    print(f"{removed} columns removed!")

    print(f"{df_raw.columns.size} columns left: {list(df_raw.columns)}\n")

    print(df_raw["num_rounds"].value_counts())

    topuria_mask = (df_raw["f_1_name"] == "Ilia Topuria") | (df_raw["f_2_name"] == "Ilia Topuria")

    cols_to_show = [
      "f_1_name",
      "f_1_fighter_w",
      "f_1_fighter_l",
      "f_1_fighter_d",
      "f_2_name",
      "f_2_fighter_w",
      "f_2_fighter_l",
      "f_2_fighter_d",
      "event_date",
    ]

    topuria = df_raw[topuria_mask][cols_to_show].copy()

    print(f"{topuria.head(5)}")
