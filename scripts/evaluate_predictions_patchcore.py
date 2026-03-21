import os
import csv
from pathlib import Path


def list_files(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {f.name for f in folder.iterdir() if f.is_file()}


def safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def evaluate_patchcore():
    # Get script directory (parent path)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Build paths relative to script location
    pred_bad_dir = BASE_DIR / "prediction_outputs_patchcore" / "Patchcore" / "latest" / "images" / "bad"
    pred_good_dir = BASE_DIR / "prediction_outputs_patchcore" / "Patchcore" / "latest" / "images" / "good"
    gt_bad_dir = BASE_DIR / "dataset" / "test" / "bad"
    gt_good_dir = BASE_DIR / "dataset" / "test" / "good"

    print("Prediction bad:", pred_bad_dir)
    print("Prediction good:", pred_good_dir)
    print("GT bad:", gt_bad_dir)
    print("GT good:", gt_good_dir)

    pred_bad = list_files(pred_bad_dir)
    pred_good = list_files(pred_good_dir)
    gt_bad = list_files(gt_bad_dir)
    gt_good = list_files(gt_good_dir)

    pred_all = pred_bad | pred_good
    gt_all = gt_bad | gt_good
    common_files = sorted(pred_all & gt_all)

    tp = tn = fp = fn = 0
    results = []

    for filename in common_files:
        predicted_label = "bad" if filename in pred_bad else "good"
        actual_label = "bad" if filename in gt_bad else "good"

        if actual_label == "bad" and predicted_label == "bad":
            category = "TP"
            tp += 1
        elif actual_label == "good" and predicted_label == "good":
            category = "TN"
            tn += 1
        elif actual_label == "good" and predicted_label == "bad":
            category = "FP"
            fp += 1
        else:
            category = "FN"
            fn += 1

        results.append([filename, actual_label, predicted_label, category])

    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    sensitivity = recall
    specificity = safe_div(tn, tn + fp)
    false_positive_rate = safe_div(fp, fp + tn)
    false_negative_rate = safe_div(fn, fn + tp)

    # Save CSV
    output_csv = BASE_DIR / "patchcore_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ground_truth", "prediction", "category"])
        writer.writerows(results)

    print("\n=== Confusion Matrix ===")
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    print("\n=== Metrics ===")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("False Positive Rate:", false_positive_rate)
    print("False Negative Rate:", false_negative_rate)

    print("\nResults saved to:", output_csv)


if __name__ == "__main__":
    evaluate_patchcore()