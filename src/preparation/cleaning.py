from pathlib import Path
from typing import Dict, List

import pandas as pd


def recalculate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate the *W-L-D* records obtained by the fighters up to the match date.

    This function prevents lookahead bias, a type of data leakage, ensuring that
    a fighter's records only reflect their past history as of the match date.

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
        else:  # Draw
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


def clean_result_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the "results" feature values
    by mapping the values to a restricted domain.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    df = df[df["result"] != "Could Not Continue"]
    df = df[df["result"] != "DQ"]

    cleanup_map: Dict[str, str] = {
        "TKO - Doctor's Stoppage": "KO/TKO",
        "Decision - Split": "Decision",
        "Decision - Unanimous": "Decision",
    }

    df.loc[:, "result"] = df["result"].replace(cleanup_map)
    return df


def recalculate_ko_and_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculates the number of wins and losses by KO or submission
    for each fighter up to the date of the match.

    :param df: The dataframe.
    :return: The updated dataframe. Eight features will be added:
        f1_ko_w, f1_ko_l, f2_ko_w, f2_ko_l, f1_sub_w, f1_sub_l, f2_sub_w, f2_sub_l.
    """
    df = df.copy()

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    df["match_id"] = df.index

    f1_data = df[["match_id", "event_date", "f1_name", "result", "winner"]].copy()
    f1_data.columns = ["match_id", "event_date", "fighter", "result", "winner"]
    f1_data["is_f1"] = True

    f2_data = df[["match_id", "event_date", "f2_name", "result", "winner"]].copy()
    f2_data.columns = ["match_id", "event_date", "fighter", "result", "winner"]
    f2_data["is_f1"] = False

    tall = pd.concat([f1_data, f2_data], ignore_index=True)

    tall["is_winner"] = (tall["fighter"] == tall["winner"]).astype(int)

    methods_map = {"KO/TKO": "ko", "Submission": "sub"}

    for m, n in methods_map.items():
        tall[f"{n}_w"] = ((tall["is_winner"] == 1) & (tall["result"] == m)).astype(int)
        tall[f"{n}_l"] = ((tall["is_winner"] == 0) & (tall["result"] == m)).astype(int)

    tall = tall.sort_values(["fighter", "event_date", "match_id"])

    cols_to_sum = [f"{n}_w" for n in methods_map.values()] + [
        f"{n}_l" for n in methods_map.values()
    ]

    stats_history = (
        tall.groupby("fighter")[cols_to_sum]
        .apply(lambda x: x.expanding().sum().shift(1).fillna(0))
        .reset_index(level=0, drop=True)
    )

    tall_results = pd.concat([tall[["match_id", "is_f1"]], stats_history], axis=1)

    f1_stats = (
        tall_results[tall_results["is_f1"]].drop(columns="is_f1").set_index("match_id")
    )
    f1_stats.columns = [f"f1_{c}" for c in f1_stats.columns]

    f2_stats = (
        tall_results[~tall_results["is_f1"]].drop(columns="is_f1").set_index("match_id")
    )
    f2_stats.columns = [f"f2_{c}" for c in f2_stats.columns]

    df = df.merge(f1_stats, left_on="match_id", right_index=True, how="left")
    df = df.merge(f2_stats, left_on="match_id", right_index=True, how="left")

    return df.drop(columns="match_id")


def process_weight_class_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the "weight_class" feature values
    by mapping the values to a restricted domain.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    cleanup_map: Dict[str, str] = {
        "UFC Light Heavyweight Title": "Light Heavyweight",
        "UFC Bantamweight Title": "Bantamweight",
        "UFC Middleweight Title": "Middleweight",
        "Nieznana": "Open Weight",
    }

    df["weight_class"] = df["weight_class"].replace(cleanup_map)

    df = df[df["weight_class"] != "Open Weight"]
    df = df[df["weight_class"] != "Catch Weight"]
    return df


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw.csv")

    match_related_features = [
        "winner",
        "weight_class",
        "result",
        "gender",
        "event_date",
        "num_rounds",
    ]

    fighters_related_features = [
        "name",
        "fighter_height_cm",
        "fighter_weight_lbs",
        "fighter_reach_cm",
        "fighter_stance",
        "fighter_dob",
    ]

    df = raw_df[match_related_features].copy()

    for i in range(1, 3):
        for f in fighters_related_features:
            df[f"f{i}_{f}"] = raw_df[f"f_{i}_{f}"]

    # Remove UFC Women's matches and drop "gender" feature
    df = df[df["gender"] == "M"]
    df.drop(columns=["gender"], inplace=True)

    # Recalculate fighters' records (W-L-D) up to the match date.
    df = recalculate_records(df)

    # Clean "result" feature by mapping their values
    print("Fight ending methods before cleaning:")
    print(df["result"].value_counts())
    df = clean_result_feature(df)
    # df = df[df["result"] != "DQ"]
    print("\nFight ending methods after cleaning:")
    print(df["result"].value_counts())

    # Recalculate the number of wins and losses by KO or submission
    # for each fighter up to the date of the match.
    df = recalculate_ko_and_submissions(df)

    # Remove bouts with less than three rounds
    df = df[df["num_rounds"] >= 3]

    # Clean "weight_class" feature by mapping their values
    print("\nWeight classes:")
    print(df["weight_class"].value_counts())
    df = process_weight_class_feature(df)
    print("\nWeight classes after processing:")
    print(df["weight_class"].value_counts())

    print(f"\nDataset shape after cleaning: {df.shape}")

    print("\nSaving dataset...")
    filepath = Path("data/cleaned.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding="utf-8", index=False, header=True)
    print(f"Cleaned dataset successfully saved at {filepath}!")
