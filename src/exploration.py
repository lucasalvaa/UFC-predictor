import pandas as pd

from src.imputation import missing_values_info

if __name__ == "__main__":
    df_raw = pd.read_csv("data/raw.csv")
    print(f"Dataset shape: {df_raw.shape}")
    print(f"Memory Usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nInformations about columns:")
    print(df_raw.info())

    print(f"\nColumns list: {list(df_raw.columns)}")

    # The columns gets divided into five clusters
    all_cols = df_raw.columns.tolist()

    odds_cols = [c for c in all_cols if c.endswith("_odds")]

    round_cols = [
        c
        for c in all_cols
        if any(
            c.startswith(f"f_1_r{n}") or c.startswith(f"f_2_r{n}") for n in range(1, 6)
        )
    ]

    f1_cols = [
        c
        for c in all_cols
        if c.startswith("f_1_") and c not in odds_cols and c not in round_cols
    ]

    f2_cols = [
        c
        for c in all_cols
        if c.startswith("f_2_") and c not in odds_cols and c not in round_cols
    ]

    event_cols = [
        c for c in all_cols if not (c.startswith("f_1_") or c.startswith("f_2_"))
    ]

    print(f"\nTotal columns: {len(all_cols)}")
    print(f"Event/Bout: {len(event_cols)}")
    print(f"Fighter 1: {len(f1_cols)}")
    print(f"Fighter 2: {len(f2_cols)}")
    print(f"Rounds: {len(round_cols)}")
    print(f"Odds: {len(odds_cols)}")

    print(f"\n{len(f1_cols)} fighters-related features: {[c[4:] for c in f1_cols]}")
    assert len(event_cols) + len(f1_cols) + len(f2_cols) + len(round_cols) + len(
        odds_cols
    ) == len(all_cols)

    event_cols.remove("result_details")
    event_cols.remove("finish_round")
    event_cols.remove("finish_time")

    f1_cols = [
        c
        for c in f1_cols
        if c.startswith("f_1_fighter_") or c.endswith("name") or c.endswith("url")
    ]

    f2_cols = [
        c
        for c in f2_cols
        if c.startswith("f_2_fighter_") or c.endswith("name") or c.endswith("url")
    ]

    print(f"{len(event_cols)} event-related features: {event_cols}")
    print(f"{len(f1_cols)} fighters-related features: {[c[4:] for c in f1_cols]}")

    filtered = df_raw[event_cols + f1_cols + f2_cols].copy()

    missing_values_info(filtered[event_cols])
    missing_values_info(filtered[f1_cols])
    missing_values_info(filtered[f2_cols])

    # Ignoring rows with the highest percentage of null values
    values = {
        "f_1_fighter_nc_dq": 0,
        "f_2_fighter_nc_dq": 0,
        "f_1_fighter_nickname": "",
        "f_2_fighter_nickname": "",
    }
    filtered.fillna(value=values, inplace=True)

    mv_records = filtered.shape[0] - filtered.dropna().shape[0]
    print(f"\n{mv_records} records with missing values")

    # Drop the feature event_state
    filtered.drop(columns="event_state", inplace=True)
    mv_records = filtered.shape[0] - filtered.dropna().shape[0]
    print(f"\n{mv_records} records with missing values")

    # Data quality check of some features
    print(filtered["num_rounds"].value_counts())
    print(filtered["gender"].value_counts())
    print(filtered["weight_class"].value_counts())
    print(filtered["result"].value_counts())

    # Visualizing Topuria's matches
    print(f"Date range: {filtered.event_date.min()} to {filtered.event_date.max()}")
    topuria = "Ilia Topuria".strip().lower()
    bouts = []

    for b in filtered.itertuples():
        if (
            topuria in b.f_1_name.strip().lower()
            or topuria in b.f_2_name.strip().lower()
        ):
            bouts.append(b)

    print(f"\nBouts involving Ilia Topuria found: {len(bouts)}")

    for b in bouts:
        if topuria in b.f_1_name.strip().lower():
            print(
                f"Topuria's record and SlpM updated to the match of {b.event_date}: "
                f"{b.f_1_fighter_w}-{b.f_1_fighter_l}-{b.f_1_fighter_d}, "
                f"{b.f_1_fighter_SlpM}"
            )
        if topuria in b.f_2_name.strip().lower():
            print(
                f"Topuria's record and SlpM updated to the match of {b.event_date}: "
                f"{b.f_2_fighter_w}-{b.f_2_fighter_l}-{b.f_2_fighter_d}, "
                f"{b.f_2_fighter_SlpM}"
            )
