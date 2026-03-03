import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Paths

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.pkl"
CONFIG_PATH = ROOT / "models" / "config.json"


# Model Loader (Load Once)

def load_model() -> Tuple[object, Dict]:
    """
    Load trained model and configuration.
    """
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, config


# Load model globally (so it doesn't reload every prediction)
MODEL, CONFIG = load_model()

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# Validation

def validate_features(features: Dict) -> None:
    """
    Ensure required feature keys exist.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


# Prediction Function

def predict_one(features: Dict) -> Dict:
    """
    Predict fraud probability and decision for a single transaction.
    """

    validate_features(features)

    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=float)

    prob = float(MODEL.predict_proba(x)[0, 1])
    threshold = float(CONFIG["threshold"])
    pred = int(prob >= threshold)

    return {
        "fraud_probability": prob,
        "prediction": pred,
        "threshold": threshold,
    }

# Local Testing

if __name__ == "__main__":
    dummy = {col: 0.0 for col in FEATURE_COLUMNS}
    result = predict_one(dummy)
    print(result)