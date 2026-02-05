from pathlib import Path
from typing import Dict, List

import pandas as pd


def extract_age(df: pd.DataFrame) -> int:
    """Add *f1_age* and *f2_age* features to the dataframe,
    removing the records where the fighter's date of birth is null.

    :param df: The dataframe.
    :return: The number of records removed.
    """

    event_datetime = pd.to_datetime(df["event_date"])

    f1_dob_datetime = pd.to_datetime(df["f1_fighter_dob"])
    f2_dob_datetime = pd.to_datetime(df["f2_fighter_dob"])

    df["f1_age"] = (event_datetime - f1_dob_datetime).dt.days / 365.25
    df["f2_age"] = (event_datetime - f2_dob_datetime).dt.days / 365.25

    len_before = len(df)
    df.dropna(subset=["f1_age", "f2_age"], inplace=True)
    return len_before - len(df)

def recalculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate the *W-L-D* statistics obtained by the fighters up to the match date.

    This function prevents lookahead bias, a type of data leakage, ensuring that
    a fighter's statistics only reflect their past history as of the match date.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    name_cols: List[str] = ["f1_name", "f2_name", "winner"]
    for col in name_cols:
        df[col] = df[col].astype(str).str.strip()

    # Records are sorted in ascending chronological time
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(by="event_date").reset_index(drop=True)

    # fighter_stats: { "Name": [W, L, D] }
    fighter_stats: Dict[str, List[int]] = {}

    # Lists to collect the data to be inserted in the DF at the end of the function
    f1_w, f1_l, f1_d = [], [], []
    f2_w, f2_l, f2_d = [], [], []

    for _, row in df.iterrows():
        f1: str = row["f1_name"]
        f2: str = row["f2_name"]
        winner_name: str = row["winner"]

        # Initialization of never-before-seen fighters
        if f1 not in fighter_stats:
            fighter_stats[f1] = [0, 0, 0]
        if f2 not in fighter_stats:
            fighter_stats[f2] = [0, 0, 0]

        # Assign the statistics obtained up to that match
        f1_w.append(fighter_stats[f1][0])
        f1_l.append(fighter_stats[f1][1])
        f1_d.append(fighter_stats[f1][2])

        f2_w.append(fighter_stats[f2][0])
        f2_l.append(fighter_stats[f2][1])
        f2_d.append(fighter_stats[f2][2])

        # Update the statistics in the dictionary based on the match result
        if winner_name == f1:
            fighter_stats[f1][0] += 1
            fighter_stats[f2][1] += 1
        elif winner_name == f2:
            fighter_stats[f2][0] += 1
            fighter_stats[f1][1] += 1
        else: # Draw
            fighter_stats[f1][2] += 1
            fighter_stats[f2][2] += 1

    # DataFrame update
    df["f1_fighter_w"] = f1_w
    df["f1_fighter_l"] = f1_l
    df["f1_fighter_d"] = f1_d
    df["f2_fighter_w"] = f2_w
    df["f2_fighter_l"] = f2_l
    df["f2_fighter_d"] = f2_d

    return df



if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw.csv")

    match_features = ["winner", "weight_class", "gender", "event_date"]

    fighter_features = [
        "name",
        "fighter_w",
        "fighter_l",
        "fighter_d",
        "fighter_height_cm",
        "fighter_weight_lbs",
        "fighter_reach_cm",
        "fighter_stance",
        "fighter_dob",
    ]

    df = raw_df[match_features]

    for i in range(1, 3):
        for f in fighter_features:
            df[f"f{i}_{f}"] = raw_df[f"f_{i}_{f}"]

    print(f"Columns extracted: {list(df.columns)}")

    df = recalculate_statistics(df)
    extract_age(df)

    filepath = Path("data/processed/extracted.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding="utf-8", index=False, header=True)
