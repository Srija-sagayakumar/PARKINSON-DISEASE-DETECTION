from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parkinson_tremor.data import load_dataset  # noqa: E402
from src.parkinson_tremor.inference import IMU_COLUMNS, load_trained_model, predict_from_dataframe  # noqa: E402
from src.parkinson_tremor.ml_models import train_and_evaluate_ml_models  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "artifacts/best_ml_model.joblib"
DEFAULT_TRAIN_DATASET = PROJECT_ROOT / "uploads/imu-hand-tremor-parkinsons.csv"

st.set_page_config(page_title="Parkinson Tremor Predictor", layout="centered")
st.title("Parkinson Tremor Prediction")
st.write("Upload an IMU CSV file with accelerometer and gyroscope columns.")
st.code(", ".join(IMU_COLUMNS))


def ensure_model() -> object | None:
    if MODEL_PATH.exists():
        return load_trained_model(MODEL_PATH)

    st.warning(
        "Trained model not found. Click the button below to train it, "
        "or train manually from terminal."
    )
    st.code(
        'python main.py --data "uploads/imu-hand-tremor-parkinsons.csv" --train-ml',
        language="bash",
    )

    if st.button("Train model now"):
        if not DEFAULT_TRAIN_DATASET.exists():
            st.error(f"Training dataset not found: {DEFAULT_TRAIN_DATASET}")
            st.stop()

        with st.spinner("Training ML models. This can take a minute..."):
            dataset = load_dataset(DEFAULT_TRAIN_DATASET)
            output_dir = PROJECT_ROOT / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)
            train_and_evaluate_ml_models(dataset, output_dir, cv_folds=3)

        st.success("Training complete. Reloading model...")
        return load_trained_model(MODEL_PATH)

    st.stop()


def render_diagnosis(result) -> None:
    st.subheader("Prediction Summary")

    if result.parkinsons_detected is True:
        st.error("Parkinson's disease detected: YES")
    elif result.parkinsons_detected is False:
        st.success("Parkinson's disease detected: NO")
    else:
        st.warning("Parkinson's disease detected: INCONCLUSIVE")

    st.write(result.diagnosis_text)

    normal_pct = result.condition_percentages["Normal"]
    mild_pct = result.condition_percentages["Mild Tremor"]
    parkinson_pct = result.condition_percentages["Parkinson's Disease"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Normal", f"{normal_pct:.2f}%")
    col2.metric("Mild Tremor", f"{mild_pct:.2f}%")
    col3.metric("Parkinson's Disease", f"{parkinson_pct:.2f}%")

    st.caption("Percentages use model probabilities when available; otherwise window-vote percentages.")


model = ensure_model()
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head())

        result = predict_from_dataframe(df, model=model)
        render_diagnosis(result)

        st.write(f"Windows evaluated: {result.total_windows}")
        st.write("Window vote counts:")
        st.json(result.vote_counts)
        st.write("Window-by-window predictions:")
        st.write(result.window_predictions)
    except Exception as exc:
        st.error(str(exc))
