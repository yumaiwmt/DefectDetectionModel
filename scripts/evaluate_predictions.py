from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def infer_true_label(image_path: str | Path) -> int:
    """
    Infer ground-truth image label from folder name.

    Assumes:
      dataset/test/good/... -> 0
      dataset/test/bad/...  -> 1
    """
    path = Path(image_path)
    parent_name = path.parent.name.lower()

    if parent_name == "good":
        return 0
    if parent_name == "bad":
        return 1

    raise ValueError(f"Could not infer label from path: {path}")


def bucket_name(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"
    raise ValueError(f"Unexpected labels: true={true_label}, pred={pred_label}")


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    base_path = Path(__file__).resolve().parent.parent

    predictions_csv = base_path / "prediction_outputs" / "predictions.csv"
    sorted_root = base_path / "evaluation_outputs" / "sorted"
    metrics_json = base_path / "evaluation_outputs" / "summary_metrics.json"
    detailed_csv = base_path / "evaluation_outputs" / "evaluated_predictions.csv"

    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_csv}\n"
            "Run inference first and make sure it writes predictions.csv."
        )

    df = pd.read_csv(predictions_csv)

    required_cols = {"image_path", "pred_label", "anomaly_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions.csv is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["true_label"] = df["image_path"].apply(infer_true_label)

    def normalize_pred_label(value) -> int:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "anomalous", "anomaly", "bad", "defect"}:
                return 1
            if v in {"0", "normal", "good", "ok"}:
                return 0
            raise ValueError(f"Unrecognized pred_label string: {value}")
        return int(value)

    df["pred_label"] = df["pred_label"].apply(normalize_pred_label)

    df["bucket"] = [
        bucket_name(t, p) for t, p in zip(df["true_label"], df["pred_label"])
    ]

    for row in df.itertuples(index=False):
        image_path = Path(row.image_path)
        if not image_path.exists():
            print(f"Warning: image not found, skipping copy: {image_path}")
            continue

        destination = sorted_root / row.bucket / image_path.name
        safe_copy(image_path, destination)

    y_true = df["true_label"].tolist()
    y_pred = df["pred_label"].tolist()
    y_score = df["anomaly_score"].tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "num_images": int(len(df)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }

    if len(set(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv(detailed_csv, index=False)

    print("\nEvaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nDetailed CSV: {detailed_csv}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Sorted images: {sorted_root}")


if __name__ == "__main__":
    main()