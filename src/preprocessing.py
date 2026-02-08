from pathlib import Path

import pandas as pd

from src.feature_eng import (
    calculate_delta,
    extract_age,
    process_result_feature,
    process_weight_class_feature,
    recalculate_ko_and_submissions,
    recalculate_records,
    symmetrize_dataset,
)
from src.imputation import (
    impute_fighters_stance,
    impute_physical_data,
    missing_values_info,
)

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
    df = process_result_feature(df)
    print("\nFight endings after cleaning:")
    print(df["result"].value_counts())

    # Recalculate the number of wins and losses by KO or submission
    # for each fighter up to the date of the match.
    df = recalculate_ko_and_submissions(df)

    # Drop records where one or both fighters do not have a date of birth
    len_before = len(df)
    df = df.dropna(subset=["f1_fighter_dob", "f2_fighter_dob"])
    print(
        f"\n{len_before - len(df)} records without a date of birth have been dropped."
    )
    missing_values_info(df)

    # Impute null values for height, reach and stance columns
    df = impute_physical_data(df)
    df = impute_fighters_stance(df)
    missing_values_info(df)

    # Clean "weight_class" feature by mapping their values
    df.dropna(subset=["weight_class"], inplace=True)
    print("\nWeight classes:")
    print(df["weight_class"].value_counts())
    df = process_weight_class_feature(df)
    print("\nWeight classes after processing:")
    print(df["weight_class"].value_counts())

    # Calculate the age of both fighters
    print("\nExtracting fighters' ages up to the date of each match...")
    df = extract_age(df)

    # Calculate the differences between fighters' ages, heights, weights and reaches
    df = calculate_delta(target="age", feature="age", df=df, drop=False)
    df = calculate_delta(
        target="height", feature="fighter_height_cm", df=df, drop=False
    )
    df = calculate_delta(
        target="weight", feature="fighter_weight_lbs", df=df, drop=False
    )
    df = calculate_delta(target="reach", feature="fighter_reach_cm", df=df, drop=False)

    # Calculate some features related to fighters' statistics
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

    # Categorichal features to be dropped
    categorichal_cols = [
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
    ]

    # Fighters' specific features to be dropped
    fighter_spec_columns = [
        "f1_fighter_w",
        "f1_fighter_l",
        "f1_fighter_d",
        "f2_fighter_w",
        "f2_fighter_l",
        "f2_fighter_d",
        # "f1_ko_w", "f1_ko_l", "f2_ko_w", "f2_ko_l",
        # "f1_sub_w", "f1_sub_l", "f2_sub_w", "f2_sub_l",
        "f1_fighter_height_cm",
        "f2_fighter_height_cm",
        "f1_fighter_weight_lbs",
        "f2_fighter_weight_lbs",
        "f1_fighter_reach_cm",
        "f2_fighter_reach_cm",
        "f1_age",
        "f2_age",
    ]

    df = df.drop(
        columns=[
            col for col in categorichal_cols + fighter_spec_columns if col in df.columns
        ]
    )

    print("\nData preprocessing is over. Resulting features:")
    print(df.info())

    print("\nTarget variable's distribution:")
    print(df["f1_win"].value_counts())

    df_swapped = symmetrize_dataset(df)

    print("\nTarget variable's distribution after symmetrization:")
    print(df_swapped["f1_win"].value_counts())

    print("\nSaving both datasets...")

    filepath = Path("data/processed.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding="utf-8", index=False, header=True)
    print(f"Non-balanced dataset successfully saved at {filepath}!")

    filepath = Path("data/balanced.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_swapped.to_csv(filepath, encoding="utf-8", index=False, header=True)

    print(f"Balanced dataset successfully saved at {filepath}!")
