import json
import time
from typing import Dict

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scraper import search_fighter_stats

app = FastAPI(title="UFC Predictor API")

# Loading artifacts
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
with open("model_columns.json", "r") as f:
    MODEL_COLUMNS = json.load(f)


class FighterData(BaseModel):
    """Information extracted per fighter from the scraper."""

    name: str
    stance: str
    height_cm: float
    weight_lbs: float
    reach_cm: float
    age: float
    wins: int
    losses: int
    draws: int
    wins_ko: int
    wins_sub: int
    losses_ko: int
    losses_sub: int


class PredictionRequest(BaseModel):
    """Body of the HTTP request."""

    f1_name: str
    f2_name: str
    rounds: int = 3


@app.get("/")
def home() -> Dict[str, str]:
    """Set a message for the base URL."""
    return {"message": "UFC Predictor is Live! Use /predict"}


def scrape_fighter(fighter_name: str) -> Dict[str, str | float | int]:
    """Extract fighter information from *ufcstats.com*."""
    data = search_fighter_stats(fighter_name)
    if not data:
        raise HTTPException(status_code=404, detail="Fighter not found")

    return data


def compute_information(f1: FighterData, f2: FighterData, rounds: int) -> pd.DataFrame:
    """Prepare a one-row dataframe with the information needed for prediction.

    :param f1: Red-corner fighter
    :param f2: Blue-corner fighter
    :param rounds: Rounds of the match
    :return: A one-row dataframe with the information needed for prediction
    """
    thresholds = [125, 135, 145, 155, 170, 185, 205, 265]  # weight classes thresholds
    weight_class_id = next(
        (i for i, t in enumerate(thresholds) if f1.weight_lbs <= t), -1
    )

    total_fights: Dict[str, float] = {
        "f1": f1.wins + f1.losses + f1.draws,
        "f2": f2.wins + f2.losses + f2.draws,
    }

    input_data = {
        "delta_height": f1.height_cm - f2.height_cm,
        "delta_weight": f1.weight_lbs - f2.weight_lbs,
        "delta_reach": f1.reach_cm - f2.reach_cm,
        "delta_age": f1.age - f2.age,
        "same_stance": int(f1.stance == f2.stance),
        "f1_ko_w": f1.wins_ko,
        "f2_ko_w": f2.wins_ko,
        "f1_ko_l": f1.losses_ko,
        "f2_ko_l": f2.losses_ko,
        "f1_sub_w": f1.wins_sub,
        "f2_sub_w": f2.wins_sub,
        "f1_sub_l": f1.losses_sub,
        "f2_sub_l": f2.losses_sub,
        "delta_experience": total_fights["f1"] - total_fights["f2"],
        "delta_win_rate": (f1.wins / total_fights["f1"] * 100)
        - (f2.wins / total_fights["f2"] * 100),
        "delta_sub_threat": f1.wins_sub / f1.wins - f2.wins_sub / f2.wins,
        "delta_ko_power": f1.wins_ko / f1.wins - f2.wins_ko / f2.wins,
        "delta_chin_durability": f1.losses_ko / total_fights["f1"]
        - f2.losses_ko / total_fights["f2"],
        "num_rounds": rounds,
        "weight_class_id": weight_class_id,
        "date": int(time.time()),
    }

    df = pd.DataFrame([input_data])

    # Aligning columns
    # Missing columns are zero-filled
    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Reordering columns
    df = df[MODEL_COLUMNS]
    df = df.astype(float)
    return df


@app.post("/predict")
def predict_match(request: PredictionRequest) -> Dict[str, str]:
    """Define API endpoint method."""
    # Scraping fighters data from ufcstats.com
    stats_f1 = scrape_fighter(request.f1_name)
    stats_f2 = scrape_fighter(request.f2_name)

    f1_data = FighterData(**stats_f1)
    f2_data = FighterData(**stats_f2)

    # Calculating the features needed for prediction, e.g. delta_experience
    df = compute_information(f1_data, f2_data, request.rounds)

    # Feature scaling
    scaled_data = scaler.transform(df)
    df = pd.DataFrame(scaled_data, columns=MODEL_COLUMNS)

    # Prediction
    prob = model.predict_proba(df)[0]

    winner = request.f1_name if prob[1] > prob[0] else request.f2_name
    confidence = prob[1] if prob[1] > prob[0] else prob[0]

    return {"prediction": winner, "confidence": f"{confidence:.2%}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
