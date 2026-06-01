from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import extract_features_from_signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts/best_ml_model.joblib"
IMU_COLUMNS = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]
DEFAULT_SAMPLING_RATE = 100.0
DEFAULT_WINDOW_SIZE = 150
DEFAULT_STEP_SIZE = 150

CLASS_NAME_MAP = {
    1: "Normal",
    2: "Mild Tremor",
    3: "Parkinson's Disease",
}


@dataclass
class PredictionResult:
    predicted_class: int | None
    predicted_label: str
    status: str
    is_tie: bool
    top_classes: list[int]
    window_predictions: list[int]
    vote_counts: dict[int, int]
    vote_percentages: dict[int, float]
    average_probabilities: dict[int, float]
    confidence_score: float
    total_windows: int
    condition_percentages: dict[str, float]
    parkinsons_detected: bool | None
    diagnosis_text: str


def load_trained_model(model_path: Path | str = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Train first with: python main.py --data 'uploads/imu-hand-tremor-parkinsons.csv' --train-ml"
        )
    return joblib.load(model_path)


def read_signal_csv(file_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [column for column in IMU_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV must contain these columns: {IMU_COLUMNS}. Missing: {missing}"
        )
    return df


def create_windows(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> list[np.ndarray]:
    values = df[IMU_COLUMNS].to_numpy(dtype=float)
    if len(values) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows in the uploaded signal, but received {len(values)}."
        )

    windows: list[np.ndarray] = []
    for start in range(0, len(values) - window_size + 1, step_size):
        windows.append(values[start : start + window_size])
    return windows


def build_inference_features(
    df: pd.DataFrame,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> pd.DataFrame:
    windows = create_windows(df, window_size=window_size, step_size=step_size)
    rows = [extract_features_from_signal(window, sampling_rate) for window in windows]
    return pd.DataFrame(rows)


def predict_from_dataframe(
    df: pd.DataFrame,
    model=None,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> PredictionResult:
    if model is None:
        model = load_trained_model()

    feature_table = build_inference_features(
        df,
        sampling_rate=sampling_rate,
        window_size=window_size,
        step_size=step_size,
    )
    predictions = model.predict(feature_table)
    probability_map = compute_average_probabilities(model, feature_table)
    values, counts = np.unique(predictions, return_counts=True)

    vote_counts = {int(label): int(count) for label, count in zip(values, counts)}
    vote_percentages = {
        int(label): float(count / len(predictions) * 100.0) for label, count in zip(values, counts)
    }

    max_count = int(np.max(counts))
    top_classes = [int(label) for label, count in zip(values, counts) if int(count) == max_count]
    is_tie = len(top_classes) > 1
    predicted_class = None if is_tie else int(top_classes[0])
    confidence_score = float(max_count / len(predictions) * 100.0)

    source_percentages = probability_map if probability_map else vote_percentages
    condition_percentages = build_condition_percentages(source_percentages)
    parkinsons_detected = determine_parkinsons_detection(predicted_class, is_tie)
    diagnosis_text = build_diagnosis_text(predicted_class, is_tie)

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_label=class_label_text(predicted_class),
        status="Inconclusive" if is_tie else "Predicted",
        is_tie=is_tie,
        top_classes=top_classes,
        window_predictions=[int(value) for value in predictions.tolist()],
        vote_counts=vote_counts,
        vote_percentages=vote_percentages,
        average_probabilities=probability_map,
        confidence_score=confidence_score,
        total_windows=len(predictions),
        condition_percentages=condition_percentages,
        parkinsons_detected=parkinsons_detected,
        diagnosis_text=diagnosis_text,
    )


def predict_from_csv(
    file_path: Path | str,
    model=None,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> PredictionResult:
    df = read_signal_csv(file_path)
    return predict_from_dataframe(
        df,
        model=model,
        sampling_rate=sampling_rate,
        window_size=window_size,
        step_size=step_size,
    )


def compute_average_probabilities(model, feature_table: pd.DataFrame) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(feature_table)
    if probabilities.ndim != 2:
        return {}

    classes = [int(value) for value in model.classes_.tolist()]
    mean_probabilities = probabilities.mean(axis=0)
    return {label: float(prob * 100.0) for label, prob in zip(classes, mean_probabilities)}


def build_condition_percentages(source: dict[int, float]) -> dict[str, float]:
    return {
        "Normal": float(source.get(1, 0.0)),
        "Mild Tremor": float(source.get(2, 0.0)),
        "Parkinson's Disease": float(source.get(3, 0.0)),
    }


def determine_parkinsons_detection(predicted_class: int | None, is_tie: bool) -> bool | None:
    if is_tie:
        return None
    return predicted_class == 3


def build_diagnosis_text(predicted_class: int | None, is_tie: bool) -> str:
    if is_tie:
        return "Inconclusive: mixed pattern across classes."
    if predicted_class == 1:
        return "No Parkinson's disease pattern detected (Normal)."
    if predicted_class == 2:
        return "Mild tremor pattern detected. Parkinson's disease not the dominant class."
    if predicted_class == 3:
        return "Parkinson's disease pattern detected."
    return "Unknown prediction class."


def class_label_text(predicted_class: int | None) -> str:
    if predicted_class is None:
        return "Inconclusive / Mixed pattern"
    return CLASS_NAME_MAP.get(predicted_class, f"Class {predicted_class}")
