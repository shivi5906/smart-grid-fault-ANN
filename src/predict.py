import numpy as np
import tensorflow as tf
import pickle
from dataclasses import dataclass

# ── Load once at import (FastAPI will import this module) ─
model     = tf.keras.models.load_model("models/ann_model.keras")
scaler    = pickle.load(open("models/scaler.pkl",      "rb"))
threshold = pickle.load(open("models/threshold.pkl",   "rb"))
FEATURES  = pickle.load(open("models/feature_names.pkl", "rb"))

# ── Warning thresholds ────────────────────────────────────
# GREEN  : prob < threshold          → stable
# YELLOW : threshold ≤ prob < 0.75   → borderline
# RED    : prob ≥ 0.75               → fault
YELLOW_UPPER = 0.75

@dataclass
class PredictionResult:
    probability: float        # raw sigmoid output
    status: str               # STABLE / WARNING / FAULT
    level: int                # 0 / 1 / 2  (for frontend color logic)
    threshold_used: float


def predict(input_data: list | np.ndarray) -> PredictionResult:
    """
    Args:
        input_data: 12 raw (unscaled) feature values
                    order must match FEATURES list
    Returns:
        PredictionResult dataclass
    """
    arr = np.array(input_data, dtype=np.float32).reshape(1, -1)

    if arr.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, got {arr.shape[1]}")

    arr_scaled = scaler.transform(arr)
    prob = float(model.predict(arr_scaled, verbose=0)[0][0])

    if prob < threshold:
        status, level = "STABLE",  0
    elif prob < YELLOW_UPPER:
        status, level = "WARNING", 1
    else:
        status, level = "FAULT",   2

    return PredictionResult(
        probability=round(prob, 6),
        status=status,
        level=level,
        threshold_used=threshold
    )


def predict_batch(input_matrix: np.ndarray) -> list[PredictionResult]:
    """For rolling window / simulation batch calls."""
    arr = np.array(input_matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr_scaled = scaler.transform(arr)
    probs = model.predict(arr_scaled, verbose=0).ravel()

    return [
        PredictionResult(
            probability=round(float(p), 6),
            status="STABLE"  if p < threshold    else
                   "WARNING" if p < YELLOW_UPPER else "FAULT",
            level=  0        if p < threshold    else
                    1        if p < YELLOW_UPPER else 2,
            threshold_used=threshold
        )
        for p in probs
    ]


# ── Quick smoke test ──────────────────────────────────────
if __name__ == "__main__":
    sample = [2.959060, 3.079885, 8.381025, 9.780754,
              3.763085, -0.782604, -1.257395, -1.723086,
              0.650456, 0.859578, 0.887445, 0.958034]

    result = predict(sample)
    print(f"Probability : {result.probability:.4f}")
    print(f"Status      : {result.status}")
    print(f"Level       : {result.level}")
    print(f"Threshold   : {result.threshold_used:.4f}")