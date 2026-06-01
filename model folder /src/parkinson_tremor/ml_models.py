from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import TremorDataset
from .features import build_feature_table


def get_models() -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline(
            [("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))]
        ),
        "SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=42))]
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]
        ),
    }


def train_and_evaluate_ml_models(dataset: TremorDataset, output_dir: Path, cv_folds: int = 3) -> None:
    features, labels = build_feature_table(dataset)
    labels_series = pd.Series(labels, name="label")
    groups_series = pd.Series(dataset.groups, name="group")
    splits = build_group_cv_splits(features, labels_series, groups_series, cv_folds=cv_folds, random_state=42)

    results: list[dict[str, float]] = []
    best_name = None
    best_score = -1.0
    best_model = None
    best_predictions = None
    best_targets = None

    for name, model in get_models().items():
        fold_accuracies: list[float] = []
        fold_macro_f1: list[float] = []
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for train_idx, test_idx in splits:
            x_train = features.iloc[train_idx].reset_index(drop=True)
            x_test = features.iloc[test_idx].reset_index(drop=True)
            y_train = labels_series.iloc[train_idx].to_numpy()
            y_test = labels_series.iloc[test_idx].to_numpy()

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            fold_accuracies.append(accuracy_score(y_test, predictions))
            fold_macro_f1.append(f1_score(y_test, predictions, average="macro"))
            all_predictions.append(predictions)
            all_targets.append(y_test)

        accuracy = float(np.mean(fold_accuracies))
        accuracy_std = float(np.std(fold_accuracies))
        macro_f1 = float(np.mean(fold_macro_f1))
        macro_f1_std = float(np.std(fold_macro_f1))
        results.append(
            {
                "model": name,
                "accuracy_mean": accuracy,
                "accuracy_std": accuracy_std,
                "macro_f1": macro_f1,
                "macro_f1_std": macro_f1_std,
                "folds": len(splits),
            }
        )

        combined_targets = np.concatenate(all_targets)
        combined_predictions = np.concatenate(all_predictions)
        report = classification_report(combined_targets, combined_predictions)
        report_path = output_dir / f"{name.lower().replace(' ', '_')}_report.txt"
        report_path.write_text(report, encoding="utf-8")
        save_confusion_outputs(
            output_dir=output_dir,
            model_name=name,
            y_true=combined_targets,
            y_pred=combined_predictions,
        )

        if accuracy > best_score:
            best_name = name
            best_score = accuracy
            best_model = model
            best_predictions = combined_predictions
            best_targets = combined_targets

    pd.DataFrame(results).sort_values("accuracy_mean", ascending=False).to_csv(
        output_dir / "ml_results.csv", index=False
    )

    if best_model is not None and best_name is not None:
        best_model.fit(features, labels)
        joblib.dump(best_model, output_dir / "best_ml_model.joblib")
        (output_dir / "best_model_summary.txt").write_text(
            f"Samples: {len(labels)}\n"
            f"Features: {features.shape[1]}\n"
            f"Classes: {sorted(set(labels.tolist()))}\n"
            f"Groups: {len(set(groups_series.tolist()))}\n"
            f"CV folds: {len(splits)}\n"
            f"Best model: {best_name}\n"
            f"Mean CV accuracy: {best_score:.4f}\n"
            f"Mean CV macro F1: {f1_score(best_targets, best_predictions, average='macro'):.4f}\n",
            encoding="utf-8",
        )


def build_group_cv_splits(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    cv_folds: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    group_labels = labels.groupby(groups).agg(lambda values: values.mode().iat[0])
    min_groups_per_class = int(group_labels.value_counts().min())
    n_splits = min(cv_folds, min_groups_per_class)

    if n_splits < 2:
        raise ValueError(
            "Not enough independent groups per class for cross-validation. "
            "Try reducing --segment-size to create more split groups."
        )

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(features, labels, groups))
    except Exception:
        splits = []
        fallback = GroupShuffleSplit(n_splits=20, test_size=max(1 / n_splits, 0.2), random_state=random_state)
        for train_idx, test_idx in fallback.split(features, labels, groups):
            train_labels = set(labels.iloc[train_idx].tolist())
            test_labels = set(labels.iloc[test_idx].tolist())
            if train_labels == test_labels:
                splits.append((train_idx, test_idx))
            if len(splits) == n_splits:
                break

    if len(splits) < n_splits:
        raise ValueError(
            "Could not create enough valid group-based folds. "
            "Try reducing --segment-size to create more split groups."
        )

    return splits


def save_confusion_outputs(output_dir: Path, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    stem = model_name.lower().replace(" ", "_")
    matrix_df.to_csv(output_dir / f"{stem}_confusion_matrix.csv")

    per_class_lines = []
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        fn = int(matrix[idx, :].sum() - tp)
        fp = int(matrix[:, idx].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class_lines.append(
            "\n".join(
                [
                    f"Class {label}",
                    f"TP: {tp}",
                    f"TN: {tn}",
                    f"FP: {fp}",
                    f"FN: {fn}",
                    f"Sensitivity/Recall: {sensitivity:.4f}",
                    f"Specificity: {specificity:.4f}",
                ]
            )
        )

    (output_dir / f"{stem}_confusion_metrics.txt").write_text(
        "\n\n".join(per_class_lines) + "\n",
        encoding="utf-8",
    )
