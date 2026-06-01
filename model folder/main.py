from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parkinson_tremor.data import load_dataset
from src.parkinson_tremor.ml_models import train_and_evaluate_ml_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parkinson tremor detection pipeline")
    parser.add_argument("--data", type=Path, help="Path to CSV dataset", required=True)
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=100.0,
        help="Sensor sampling rate in Hz for FFT feature extraction",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=150,
        help="Sliding window size for row-based IMU datasets",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=150,
        help="Sliding window step size for row-based IMU datasets",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=450,
        help="Non-overlapping segment size used to keep neighboring windows in the same split group",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of group-based cross-validation folds",
    )
    parser.add_argument("--train-ml", action="store_true", help="Train classical ML models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save models and reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.data,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        step_size=args.step_size,
        segment_size=args.segment_size,
    )

    if args.train_ml:
        train_and_evaluate_ml_models(dataset, args.output_dir, cv_folds=args.cv_folds)
    else:
        raise SystemExit("Use --train-ml to train the classifiers")


if __name__ == "__main__":
    main()
