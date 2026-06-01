# Parkinson Tremor Detection Project

This project implements a Parkinson tremor detection pipeline using:

- Signal processing: Butterworth filtering and FFT
- Feature-based machine learning: Random Forest, SVM, KNN, Logistic Regression

## Project Structure

- `main.py` - entry point for training and evaluation
- `streamlit_app.py` - Streamlit UI for prediction
- `src/parkinson_tremor/config.py` - project settings
- `src/parkinson_tremor/data.py` - dataset loading and IMU window generation
- `src/parkinson_tremor/signal_processing.py` - filtering and FFT utilities
- `src/parkinson_tremor/features.py` - feature extraction
- `src/parkinson_tremor/ml_models.py` - classical ML training and comparison
- `src/parkinson_tremor/inference.py` - model loading and inference helpers

## Supported Data Format (IMU)

Training and inference expect these columns:

- `xAcc`, `yAcc`, `zAcc`
- `xGyro`, `yGyro`, `zGyro`
- `label` (training dataset only)

Your dataset `uploads/imu-hand-tremor-parkinsons.csv` already matches this format.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train on the IMU dataset (this creates `artifacts/best_ml_model.joblib`):

```bash
python main.py --data "uploads/imu-hand-tremor-parkinsons.csv" --train-ml
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

## Prediction Upload Format (Streamlit)

For uploaded CSV prediction, include exactly:

- `xAcc`
- `yAcc`
- `zAcc`
- `xGyro`
- `yGyro`
- `zGyro`

Minimum rows needed with current settings: `150`.
