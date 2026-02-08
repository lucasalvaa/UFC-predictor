import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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

    cmh = sns.heatmap(lower, annot=True, cmap="coolwarm")
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


def tune_individual_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> List[Tuple[str, object]]:
    """Perform hyperparameter tuning for RF, LGBM, and XGBoost using RandomizedSearchCV.

    :param X_train: The training set.
    :param y_train: The target variable of the training set.
    :return: A list of the best fitted estimators.
    """
    # Define parameter grids for each model
    param_grids = {
        "rf": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [200, 500, 700],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
        },
        "lgbm": {
            "model": LGBMClassifier(
                random_state=42, n_jobs=1, verbose=-1, importance_type="gain"
            ),
            "params": {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31],
                "max_depth": [5, 10],
                "bagging_fraction": [0.7, 0.8],
                "feature_fraction": [0.7, 0.8],
                "min_child_samples": [20, 50],
            },
        },
        "xgb": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {
                "n_estimators": [300, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 9],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8],
            },
        },
    }

    best_estimators = []

    for name, config in param_grids.items():
        print(f"Tuning {name} model...")

        # Search across 10 different combinations, using 5-fold cross validation
        search = RandomizedSearchCV(
            config["model"],
            param_distributions=config["params"],
            n_iter=10,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train, y_train)

        print(f"Best params for {name} model: {search.best_params_}")
        best_estimators.append((name, search.best_estimator_))

    return best_estimators


def build_ensemble(
    estimators: List[Tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> object:
    """Fit the VotingClassifier on the training data using the best parameters,
    evaluate the ensemble on the test set and visualizes the results
    via a Confusion Matrix.

    :param estimators: Best parameters found fine-tuning the models.
    :param X_train: The training set.
    :param y_train: The target variable of the training set.
    :param X_test: The test set.
    :param y_test: The target variable of the test set.
    :return: The final ensemble.
    """
    print("\nInitializing Ensemble (Soft Voting)...")
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    # Fitting the Ensemble
    print("Final fitting of the ensemble using best parameters...")
    ensemble.fit(X_train, y_train)

    # Making Predictions
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Metrics
    print(f"\nFINAL ENSEMBLE ACCURACY: {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    # This helps understand if the model is biased toward a specific fighter (f1 vs f2)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fighter 2 Wins", "Fighter 1 Wins"],
        yticklabels=["Fighter 2 Wins", "Fighter 1 Wins"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Ensemble Confusion Matrix")
    plt.show()

    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", type=bool, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv("data/balanced.csv" if args.balanced else "data/processed.csv")

    TARGET = "f1_win"
    features_to_ignore = []

    correlation_matrix(df, n=10, show=False)

    print("\nBalanced dataset:" if args.balanced else "Non-balanced dataset:")

    # Data are split in train and test set
    X = df.drop(columns=[c for c in [*features_to_ignore, TARGET] if c in df.columns])
    y = df[TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scaling features while maintaining DataFrame structure for feature names
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

    # Step 1: Tune individual models
    models_best_params = tune_individual_models(X_train, y_train)

    # Step 2: Evaluate each tuned model on the test set
    results = {}
    for name, model in models_best_params:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Tuned {name} Accuracy: {acc:.4f}")

    # Step 3: Ensemble
    ensemble = build_ensemble(models_best_params, X_train, y_train, X_test, y_test)

    # Saving ensemble and scaler
    out_dir = Path("outs")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(ensemble, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # Saving the exact order of the columns used in the train
    model_columns = list(X_train.columns)
    with open(out_dir / "model_columns.json", "w") as f:
        json.dump(model_columns, f)
