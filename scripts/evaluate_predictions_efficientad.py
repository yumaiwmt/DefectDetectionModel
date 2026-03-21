from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def true_label(image_path: str | Path) -> int:
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

def pred_label(folder_name: str) -> int:
    """
    Infer predicted label from prediction output folder name.

    Assumes:
      prediction_outputs/good/... -> 0
      prediction_outputs/bad/...  -> 1
    """
    folder_name = folder_name.lower()

    if folder_name == "good":
        return 0
    if folder_name == "bad":
        return 1

    raise ValueError(f"Unexpected prediction folder: {folder_name}")


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

    sorted_root = base_path / "evaluation_outputs" / "sorted"
    metrics_json = base_path / "evaluation_outputs" / "summary_metrics.json"
    detailed_csv = base_path / "evaluation_outputs" / "evaluated_predictions.csv"

    prediction_images_root = base_path / "prediction_outputs" / "EfficientAd" / "latest" / "images"
    test_dataset_root = base_path / "dataset" / "test"
    
    if not prediction_images_root.exists():
        raise FileNotFoundError(
            f"Prediction images directory not found: {prediction_images_root}\n"
            "Run inference first using: python scripts/inference_efficientad.py"
        )

    # Build predictions dataframe from the directory structure
    # Predictions are organized in: prediction_outputs/EfficientAd/latest/images/{good,bad}/
    # We need to match each prediction file back to its original location in dataset/test/{good,bad}/ to get the true label
    records = []
    for pred_folder in ["good", "bad"]:
        pred_path = prediction_images_root / pred_folder
        if not pred_path.exists():
            continue
        
        pred_label_value = pred_label(pred_folder)
        for image_file in pred_path.glob("*"):
            if image_file.is_file() and image_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                filename = image_file.name
                
                # Find the original image in the test dataset to get the true label
                true_label_value = None
                original_path = None
                for true_folder in ["good", "bad"]:
                    candidate_path = test_dataset_root / true_folder / filename
                    if candidate_path.exists():
                        true_label_value = pred_label(true_folder)  # Use pred_label logic: good->0, bad->1
                        original_path = candidate_path
                        break
                
                if true_label_value is None:
                    print(f"Warning: Could not find original image for {filename}, skipping")
                    continue
                
                records.append({
                    "image_path": str(image_file),
                    "original_image_path": str(original_path),
                    "pred_label": pred_label_value,
                    "true_label": true_label_value,
                    "anomaly_score": 0.5,  # Placeholder; anomalib doesn't export scores by default
                })
    
    if not records:
        raise FileNotFoundError(
            f"No prediction images found in: {prediction_images_root}\n"
            "Run inference first using: python scripts/inference_efficientad.py"
        )

    df = pd.DataFrame(records)

    df["bucket"] = [
        bucket_name(true, prediction) for true, prediction in zip(df["true_label"], df["pred_label"])
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

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "num_images": int(len(df)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
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

    # Save confusion matrix as visualization
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Anomaly (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_image_path = metrics_json.parent / "confusion_matrix.png"
    plt.savefig(cm_image_path, dpi=100, bbox_inches='tight')
    plt.close()

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=['True Normal (0)', 'True Anomaly (1)'],
        columns=['Predicted Normal (0)', 'Predicted Anomaly (1)']
    )
    cm_csv_path = metrics_json.parent / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    df.to_csv(detailed_csv, index=False)

    metrics_csv_path = metrics_json.parent / "summary_metrics.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("\nEvaluation complete.")
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print("="*60)
    print(f"\nDetailed predictions CSV: {detailed_csv}")
    print(f"Summary metrics JSON: {metrics_json}")
    print(f"Summary metrics CSV: {metrics_csv_path}")
    print(f"Confusion matrix CSV: {cm_csv_path}")
    print(f"Confusion matrix image: {cm_image_path}")
    print(f"Sorted images by bucket: {sorted_root}")


if __name__ == "__main__":
    main()