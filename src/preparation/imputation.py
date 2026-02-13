from pathlib import Path

import pandas as pd


def missing_values_info(df: pd.DataFrame) -> None:
    """Print a summary of the missing values in the dataframe.

    :param df: The dataframe.
    """
    print("\nMissing Values:")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame(
            {
                "Missing Count": missing_data[missing_data > 0],
                "Missing Percentage": missing_percent[missing_data > 0],
            }
        )
        print(missing_df.sort_values("Missing Count", ascending=False))
    else:
        print("No missing values found!")


def impute_physical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in height and reach columns using the median value.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    df = df.copy()

    features = [
        ("f1_fighter_height_cm", "f2_fighter_height_cm"),
        ("f1_fighter_reach_cm", "f2_fighter_reach_cm"),
    ]

    for col_f1, col_f2 in features:
        # Create a temporary series that combines the values
        # of f1 and f2 with their corresponding weight class.
        stacked_data = pd.concat(
            [
                df[["weight_class", col_f1]].rename(columns={col_f1: "value"}),
                df[["weight_class", col_f2]].rename(columns={col_f2: "value"}),
            ]
        )

        # Create a map that associates the median value to each weight class
        median_map = stacked_data.groupby("weight_class")["value"].median()

        # Null values imputation
        df[col_f1] = df[col_f1].fillna(df["weight_class"].map(median_map))
        df[col_f2] = df[col_f2].fillna(df["weight_class"].map(median_map))

        global_median = stacked_data["value"].median()
        df[col_f1] = df[col_f1].fillna(global_median)
        df[col_f2] = df[col_f2].fillna(global_median)

    return df


def impute_fighters_stance(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in stance columns using the mode value.

    :param df: The dataframe.
    :return: The updated dataframe.
    """
    df = df.copy()

    # Create a temporary series that combines fighters' stance values
    all_stances = pd.concat([df["f1_fighter_stance"], df["f2_fighter_stance"]])

    # Get the global mode
    mode = all_stances.mode()[0]
    print(f"Global mode of fighters' stance: {mode}")

    # Null values imputation by mode
    df["f1_fighter_stance"] = df["f1_fighter_stance"].fillna(mode)
    df["f2_fighter_stance"] = df["f2_fighter_stance"].fillna(mode)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned.csv")

    missing_values_info(df)

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

    filepath = Path("data/nonull.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding="utf-8", index=False, header=True)
    print(f"\nDataset successfully saved at {filepath}!")
