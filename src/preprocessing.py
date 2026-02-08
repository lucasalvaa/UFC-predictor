import pandas as pd
from pathlib import Path

from feature_eng import *
from imputation import *

if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw.csv")

    match_related_features = [
        "winner",
        "weight_class",
        "result",
        "gender",
        "event_date",
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

    # Recalculate the W-L-D records obtained by the fighters up to the match date.
    df = recalculate_records(df)

    # Target variable extraction
    df["f1_win"] = (df["winner"] == df["f1_name"]).astype(int)

    # Clean "result" feature by mapping their values
    print("Fight endings before cleaning:")
    print(df["result"].value_counts())
    df = clean_results(df)
    print("\nFight endings after cleaning:")
    print(df["result"].value_counts())

    # Recalculate the number of wins and losses by KO or submission
    # for each fighter up to the date of the match.
    df = recalculate_ko_and_submissions(df)

    # Drop records where one or both fighters do not have a date of birth
    len_before = len(df)
    df = df.dropna(subset=["f1_fighter_dob", "f2_fighter_dob"])
    print(f"{len_before - len(df)} records without a date of birth have been dropped.")
    missing_values_info(df)

    # Impute null values for height, reach and stance columns
    df = impute_physical_data(df)
    df = impute_fighters_stance(df)
    missing_values_info(df)

    print("\nExtracting fighters' ages up to the date of each match...")
    removed = extract_age(df)

    print("\n Weight classes:")
    print(df["weight_class"].value_counts(dropna=False))

    print(f"\nSaving. Dataset shape: {df.shape}")
    filepath = Path("data/processed/extracted.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding="utf-8", index=False, header=True)
