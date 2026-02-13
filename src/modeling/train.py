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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tscv: TimeSeriesSplit,
    weights: np.ndarray,
) -> List[Tuple[str, object]]:
    """Perform hyperparameter tuning for RF, LGBM, and XGBoost using RandomizedSearchCV.

    :param X_train: The training set.
    :param y_train: The target variable of the training set.
    :param tscv: TimeSeriesSplit object.
    :param weights: List of weights used for tuning hyperparameters.
    :return: A list of the best fitted estimators.
    """
    # Define parameter grids for each model
    models = {
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
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
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
        "lr": {
            "model": LogisticRegression(
                random_state=42, solver="saga", l1_ratio=1, max_iter=5000
            ),
            "params": {
                "C": [0.1, 0.5, 1, 10],
                "l1_ratio": [0, 0.5, 1],
            },
        },
        "svc": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["rbf", "poly"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3],
            },
        },
    }

    models.pop("xgb")
    models.pop("svc")

    tuned_models = []

    for name, config in models.items():
        print(f"Tuning {name} model...")

        # Search across 10 different combinations,
        # using 3-fold Time Series cross-validation
        search = RandomizedSearchCV(
            config["model"],
            param_distributions=config["params"],
            n_iter=10,
            cv=tscv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )

        search.fit(X_train, y_train, sample_weight=weights)
        print(f"Best params for {name} model: {search.best_params_}")
        tuned_models.append((name, search.best_estimator_))

    return tuned_models


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
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

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
    df = pd.read_csv("data/balanced.csv")
    TARGET = "f1_win"

    # Splitting data using time series split
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    tscv = TimeSeriesSplit(n_splits=3)
    fold_results = []

    print("Training on unbalanced dataset...")
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        time_delta = X_train_raw["date"] - X_train_raw["date"].min()
        normalized_time = time_delta / time_delta.max()
        weights = np.exp(normalized_time)

        # Scaling features while maintaining DataFrame structure for feature names
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

        # Step 1: Tune individual models
        inner_tscv = TimeSeriesSplit(n_splits=3)
        models_best_params = run_training(X_train, y_train, inner_tscv, weights)

        # Step 2: Evaluate each tuned model on the test set
        current_fold_metrics = {}
        for name, model in models_best_params:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            current_fold_metrics[name] = acc
            print(f"Fold {i + 1} - {name} Accuracy: {acc:.4f}")

        # Step 3: Ensemble
        ensemble = build_ensemble(models_best_params, X_train, y_train, X_test, y_test)

        fold_results.append(current_fold_metrics)

        # Saving ensemble and scaler
        out_dir = Path("outs")
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(ensemble, out_dir / "model.joblib")
        joblib.dump(scaler, out_dir / "scaler.joblib")

        # Saving the exact order of the columns used in the train
        model_columns = list(X.columns)
        with open(out_dir / "model_columns.json", "w") as f:
            json.dump(model_columns, f)
