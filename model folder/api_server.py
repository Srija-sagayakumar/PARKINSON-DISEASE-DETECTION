from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.parkinson_tremor.inference import (
    DEFAULT_WINDOW_SIZE,
    IMU_COLUMNS,
    load_trained_model,
    predict_from_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts/best_ml_model.joblib"
WEBSITE_ROOT = Path("/Users/sasfamily/project website ")

LIVE_BUFFER_MAX = 1200
LIVE_PREDICTION_WINDOW = DEFAULT_WINDOW_SIZE  # 150 based on current model

ACC_COLUMNS = ["xAcc", "yAcc", "zAcc"]
GYRO_COLUMNS = ["xGyro", "yGyro", "zGyro"]
ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0

app = Flask(__name__, static_folder=None)
CORS(app)

model = load_trained_model(MODEL_PATH)

live_lock = Lock()
live_samples: deque[dict[str, float]] = deque(maxlen=LIVE_BUFFER_MAX)
latest_live_prediction: dict[str, Any] | None = None


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_sample(raw: dict[str, Any]) -> dict[str, float]:
    sample: dict[str, float] = {}
    for column in IMU_COLUMNS:
        if column not in raw:
            raise ValueError(f"Missing field '{column}' in sample")
        try:
            sample[column] = float(raw[column])
        except Exception as exc:
            raise ValueError(f"Invalid numeric value for '{column}'") from exc
    return sample


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in IMU_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=IMU_COLUMNS).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No valid numeric rows after preprocessing input data")
    return prepared


def _auto_unit_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    normalized = df.copy()

    acc_median_abs = float(normalized[ACC_COLUMNS].abs().median().median())
    gyro_median_abs = float(normalized[GYRO_COLUMNS].abs().median().median())

    converted_acc = acc_median_abs > 8.0
    converted_gyro = gyro_median_abs > 500.0

    if converted_acc:
        normalized[ACC_COLUMNS] = normalized[ACC_COLUMNS] / ACC_LSB_PER_G

    if converted_gyro:
        normalized[GYRO_COLUMNS] = normalized[GYRO_COLUMNS] / GYRO_LSB_PER_DPS

    return normalized, {
        "acc_lsb_to_g": converted_acc,
        "gyro_lsb_to_dps": converted_gyro,
    }


def _denoise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    denoised = df.copy()

    for column in IMU_COLUMNS:
        series = denoised[column]

        rolling_median = series.rolling(window=5, center=True, min_periods=1).median()
        abs_dev = (series - rolling_median).abs()
        mad = abs_dev.rolling(window=9, center=True, min_periods=1).median()

        # Hampel-style clipping for transient spikes
        threshold = 3.0 * 1.4826 * mad + 1e-6
        clipped = series.where(abs_dev <= threshold, rolling_median)

        # Short moving average to smooth high-frequency sensor noise
        smoothed = clipped.rolling(window=3, center=True, min_periods=1).mean()
        denoised[column] = smoothed

    return denoised


def _transform_for_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = _prepare_dataframe(df)
    normalized, conversion_flags = _auto_unit_normalize(prepared)
    filtered = _denoise_dataframe(normalized)

    info = {
        "input_rows": int(len(df)),
        "rows_after_numeric_cleanup": int(len(prepared)),
        "rows_after_transform": int(len(filtered)),
        "unit_conversion": conversion_flags,
        "noise_filter": {
            "enabled": True,
            "method": "hampel_clip_plus_moving_average",
            "hampel_window": 5,
            "mad_window": 9,
            "smooth_window": 3,
        },
    }
    return filtered.reset_index(drop=True), info


def _build_prediction_payload(result, preprocessing_info: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "predicted_class": result.predicted_class,
        "predicted_label": result.predicted_label,
        "parkinsons_detected": result.parkinsons_detected,
        "diagnosis_text": result.diagnosis_text,
        "condition_percentages": result.condition_percentages,
        "total_windows": result.total_windows,
        "vote_counts": result.vote_counts,
        "vote_percentages": result.vote_percentages,
        "updated_at": utc_iso(),
        "source": source,
        "preprocessing": preprocessing_info,
    }


def _predict_from_recent_window() -> dict[str, Any] | None:
    global latest_live_prediction

    if len(live_samples) < LIVE_PREDICTION_WINDOW:
        return None

    raw_window = list(live_samples)[-LIVE_PREDICTION_WINDOW:]
    raw_df = pd.DataFrame(raw_window)
    transformed_df, prep_info = _transform_for_inference(raw_df)

    result = predict_from_dataframe(transformed_df, model=model)
    latest_live_prediction = _build_prediction_payload(result, prep_info, source="live_window")
    return latest_live_prediction


@app.get("/")
def website_index():
    return send_from_directory(WEBSITE_ROOT, "index.html")


@app.get("/index.html")
def website_index_file():
    return send_from_directory(WEBSITE_ROOT, "index.html")


@app.get("/<path:filename>")
def website_assets(filename: str):
    website_file = WEBSITE_ROOT / filename
    if website_file.exists() and website_file.is_file():
        return send_from_directory(WEBSITE_ROOT, filename)
    return {"error": "Not Found"}, 404


@app.get("/health")
def health() -> tuple[dict, int]:
    return {"status": "ok"}, 200


@app.post("/predict")
def predict() -> tuple[dict, int]:
    if "file" not in request.files:
        return {"error": "Missing file. Send CSV as form-data key 'file'."}, 400

    file = request.files["file"]
    if not file or not file.filename:
        return {"error": "No CSV file provided."}, 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return {"error": f"Could not read CSV: {exc}"}, 400

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    missing = [column for column in IMU_COLUMNS if column not in df.columns]
    if missing:
        return {
            "error": "Invalid CSV format.",
            "required_columns": IMU_COLUMNS,
            "missing_columns": missing,
        }, 400

    try:
        transformed_df, prep_info = _transform_for_inference(df)
        result = predict_from_dataframe(transformed_df, model=model)
    except Exception as exc:
        return {"error": str(exc)}, 400

    response = _build_prediction_payload(result, prep_info, source="uploaded_file")
    return jsonify(response), 200


@app.post("/api/live-sample")
def ingest_live_sample() -> tuple[dict, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"error": "JSON body required."}, 400

    try:
        sample = _normalize_sample(payload)
    except ValueError as exc:
        return {"error": str(exc), "required_columns": IMU_COLUMNS}, 400

    with live_lock:
        live_samples.append(sample)
        prediction = _predict_from_recent_window()
        count = len(live_samples)

    return {
        "status": "ok",
        "buffer_count": count,
        "window_size": LIVE_PREDICTION_WINDOW,
        "prediction_ready": prediction is not None,
        "latest_prediction": prediction,
    }, 200


@app.post("/api/live-batch")
def ingest_live_batch() -> tuple[dict, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        return {"error": "JSON body with 'samples' list required."}, 400

    samples = payload["samples"]
    if not samples:
        return {"error": "'samples' list is empty."}, 400

    valid = 0
    with live_lock:
        for raw in samples:
            if not isinstance(raw, dict):
                continue
            try:
                sample = _normalize_sample(raw)
            except ValueError:
                continue
            live_samples.append(sample)
            valid += 1

        prediction = _predict_from_recent_window()
        count = len(live_samples)

    if valid == 0:
        return {"error": "No valid samples in batch.", "required_columns": IMU_COLUMNS}, 400

    return {
        "status": "ok",
        "accepted_samples": valid,
        "buffer_count": count,
        "window_size": LIVE_PREDICTION_WINDOW,
        "prediction_ready": prediction is not None,
        "latest_prediction": prediction,
    }, 200


@app.get("/api/config")
def api_config() -> tuple[dict, int]:
    return {
        "status": "ok",
        "required_columns": IMU_COLUMNS,
        "window_size": LIVE_PREDICTION_WINDOW,
        "live_buffer_max": LIVE_BUFFER_MAX,
        "endpoints": {
            "health": "/health",
            "predict_file": "/predict",
            "live_sample": "/api/live-sample",
            "live_batch": "/api/live-batch",
            "live_state": "/api/live-state",
            "live_reset": "/api/live-reset",
        },
    }, 200


@app.get("/api/live-state")
def get_live_state() -> tuple[dict, int]:
    with live_lock:
        recent = list(live_samples)[-300:]
        prediction = latest_live_prediction
        count = len(live_samples)

    return {
        "status": "ok",
        "buffer_count": count,
        "window_size": LIVE_PREDICTION_WINDOW,
        "prediction_ready": prediction is not None,
        "latest_prediction": prediction,
        "recent_samples": recent,
        "required_columns": IMU_COLUMNS,
    }, 200


@app.post("/api/live-reset")
def live_reset() -> tuple[dict, int]:
    global latest_live_prediction
    with live_lock:
        live_samples.clear()
        latest_live_prediction = None
    return {"status": "ok", "message": "Live buffer cleared."}, 200


if __name__ == "__main__":
    print("\nOpen website:     http://127.0.0.1:5001")
    print("API health:       http://127.0.0.1:5001/health")
    print("Live state:       http://127.0.0.1:5001/api/live-state")
    print("API config:       http://127.0.0.1:5001/api/config")
    print("Ingest endpoint:  POST http://127.0.0.1:5001/api/live-sample")
    print("Batch endpoint:   POST http://127.0.0.1:5001/api/live-batch\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
