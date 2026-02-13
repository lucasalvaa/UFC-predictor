from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def encode_weight_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Map the *weight_class* feature's values to a range of values from 0 to 7,
    from the lightest to the heaviest weight class.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    ordered_classes = [
        "Flyweight",
        "Bantamweight",
        "Featherweight",
        "Lightweight",
        "Welterweight",
        "Middleweight",
        "Light Heavyweight",
        "Heavyweight",
    ]

    encoding_map = {name: i for i, name in enumerate(ordered_classes)}

    df["weight_class_id"] = df["weight_class"].map(encoding_map)

    return df


def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Balance the dataset by swapping fighter 1 with fighter 2 in some rows.

    :param df: The unbalanced dataframe.
    :param seed: Random seed. Defaults to 42.
    :return: The symmetrized dataframe.
    """
    np.random.seed(seed)
    df_swapped = df.copy()

    # Select 50% of the records randomly and swaps the fighters
    mask = np.random.rand(len(df_swapped)) < 0.5

    f1_to_swap = ["f1_ko_w", "f1_sub_w", "f1_ko_l", "f1_sub_l"]
    f2_to_swap = ["f2_ko_w", "f2_sub_w", "f2_ko_l", "f2_sub_l"]
    df_swapped.loc[mask, f1_to_swap + f2_to_swap] = df_swapped.loc[
        mask, f2_to_swap + f1_to_swap
    ].values

    # Invert delta values of the selected records
    delta_cols: List[str] = [c for c in df.columns if c.startswith("delta_")]
    for col in delta_cols:
        df_swapped.loc[mask, col] = -df_swapped.loc[mask, col]

    # Update target variable
    df_swapped.loc[mask, "f1_win"] = 1 - df_swapped.loc[mask, "f1_win"]

    return df_swapped


def correlation_matrix(
    df: pd.DataFrame, n: int = 10, save: bool = True, show: bool = True
) -> None:
    """Compute the confusion matrix and show the ranking
    of the n feature pairs with the highest correlation.

    :param df: The dataframe.
    :param n: The number of feature pairs to show.
    :param save: Whether to save the confusion matrix in .png format.
    :param show: Whether to show the confusion matrix.
    """
    corr_matrix = df.corr().abs()

    # Extract the lower triangle excluding the diagonal
    lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))

    cmh = sns.heatmap(lower, annot=False, cmap="coolwarm")
    if show:
        plt.show()

    if save:
        filepath = Path("outs/corr_matrix.png")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        figure = cmh.get_figure()
        figure.savefig(filepath, dpi=400)

    # Transform the matrix into a series
    corr_series = lower.unstack().dropna()

    # Sort the values in descending order
    sorted_correlations = corr_series.sort_values(ascending=False)

    # Print the ranking of the feature pairs with the highest correlation
    print(f"Top {n} feature pairs by absolute correlation:")
    print("-" * 50)
    for (f1, f2), val in sorted_correlations.head(n).items():
        print(f"{val:.4f} | {f1} <-> {f2}")


if __name__ == "__main__":
    df = pd.read_csv("data/nonull.csv")

    # Refining target variable
    df["f1_win"] = (df["winner"] == df["f1_name"]).astype(int)

    # Encoding weight classes in range [0, 7]
    df = encode_weight_classes(df)

    # Converting date in Unix Timestamp format
    df["date"] = pd.to_datetime(df["event_date"]).astype("int64") // 10**9

    # Calculating the age of both fighters
    event_datetime = pd.to_datetime(df["event_date"])

    f1_dob_datetime = pd.to_datetime(df["f1_fighter_dob"])
    f2_dob_datetime = pd.to_datetime(df["f2_fighter_dob"])

    df["f1_age"] = (event_datetime - f1_dob_datetime).dt.days / 365.25
    df["f2_age"] = (event_datetime - f2_dob_datetime).dt.days / 365.25

    # Calculating the differences between fighters' ages, heights, weights and reaches
    df["delta_age"] = df["f1_age"] - df["f2_age"]
    df["delta_height"] = df["f1_fighter_height_cm"] - df["f2_fighter_height_cm"]
    df["delta_weight"] = df["f1_fighter_weight_lbs"] - df["f2_fighter_weight_lbs"]
    df["delta_reach"] = df["f1_fighter_reach_cm"] - df["f2_fighter_reach_cm"]

    # Calculating some attributes derived from fighters' statistics
    total_fights = [
        df["f1_fighter_w"] + df["f1_fighter_l"] + df["f1_fighter_d"],
        df["f2_fighter_w"] + df["f2_fighter_l"] + df["f2_fighter_d"],
    ]

    f1_win_rate = df["f1_fighter_w"] / total_fights[0].replace(0, 1) * 100
    f2_win_rate = df["f2_fighter_w"] / total_fights[1].replace(0, 1) * 100

    df["delta_win_rate"] = f1_win_rate - f2_win_rate

    df["delta_experience"] = total_fights[0] - total_fights[1]

    df["delta_sub_threat"] = (df["f1_sub_w"] / df["f1_fighter_w"].replace(0, 1)) - (
        df["f2_sub_w"] / df["f2_fighter_w"].replace(0, 1)
    )

    df["delta_ko_power"] = (df["f1_ko_w"] / df["f1_fighter_w"].replace(0, 1)) - (
        df["f2_ko_w"] / df["f2_fighter_w"].replace(0, 1)
    )

    df["delta_chin_durability"] = (df["f1_ko_l"] / total_fights[0].replace(0, 1)) - (
        df["f2_ko_l"] / total_fights[1].replace(0, 1)
    )

    df["same_stance"] = (df["f1_fighter_stance"] == df["f2_fighter_stance"]).astype(int)

    # Dropping features no longer considered useful
    features_to_drop = [
        # Categorical features
        "f1_fighter_dob",
        "f2_fighter_dob",
        "f1_name",
        "f2_name",
        "f1_fighter_stance",
        "f2_fighter_stance",
        "event_date",
        "winner",
        "result",
        "weight_class",
        # Fighters' related numerical features
        "f1_fighter_w",
        "f1_fighter_l",
        "f1_fighter_d",
        "f2_fighter_w",
        "f2_fighter_l",
        "f2_fighter_d",
        "f1_fighter_height_cm",
        "f2_fighter_height_cm",
        "f1_fighter_weight_lbs",
        "f2_fighter_weight_lbs",
        "f1_fighter_reach_cm",
        "f2_fighter_reach_cm",
        "f1_age",
        "f2_age",
    ]

    df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

    print("\nData preprocessing is over. Resulting features:")
    print(df.info())

    # Balancing the target variable
    print("\nTarget variable's distribution:")
    print(df["f1_win"].value_counts())
    df_swapped = balance_dataset(df)
    print("\nTarget variable's distribution after balancing:")
    print(df_swapped["f1_win"].value_counts())

    correlation_matrix(df, n=10, show=True)

    print("\nSaving dataset...")
    filepath = Path("data/balanced.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_swapped.to_csv(filepath, encoding="utf-8", index=False, header=True)
    print(f"Balanced dataset successfully saved at {filepath}!")
