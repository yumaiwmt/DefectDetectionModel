from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


CLASS_TO_LABEL = {
    "good": 0,
    "bad": 1,
}

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}


def class_label_from_name(name: str) -> int:
    key = name.lower()
    if key not in CLASS_TO_LABEL:
        raise ValueError(f"Unexpected class name: {name}")
    return CLASS_TO_LABEL[key]


def true_label(image_path: str | Path) -> int:
    path = Path(image_path)
    return class_label_from_name(path.parent.name)


def predicted_label_from_folder(folder_name: str) -> int:
    return class_label_from_name(folder_name)


def bucket_name(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    raise ValueError(f"Unexpected labels: true={y_true}, pred={y_pred}")


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_optional_scores(scores_csv_path: Path) -> dict[str, float]:
    """
    Optional CSV format:
      filename,score

    where score is the probability/confidence for positive class 'bad' (class 1).
    """
    if not scores_csv_path.exists():
        return {}

    df_scores = pd.read_csv(scores_csv_path)
    required_cols = {"filename", "score"}
    if not required_cols.issubset(df_scores.columns):
        raise ValueError(
            f"Optional scores CSV found at {scores_csv_path}, but it must contain "
            f"columns: {sorted(required_cols)}"
        )

    return dict(zip(df_scores["filename"].astype(str), df_scores["score"].astype(float)))


def build_resnet18_binary_classifier(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load ResNet18 checkpoint.

    Supports either:
    - raw state_dict
    - dict with key 'model_state_dict'
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = build_resnet18_binary_classifier(num_classes=2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_preprocess_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def softmax_score_for_positive_class(logits: torch.Tensor, positive_class: int = 1) -> float:
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, positive_class].item())


def predict_with_model(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> tuple[int, float]:
    transform = get_preprocess_transform()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred = int(torch.argmax(logits, dim=1).item())
        score_bad = softmax_score_for_positive_class(logits, positive_class=1)

    return pred, score_bad


def generate_gradcam_overlay(
    model: nn.Module,
    image_path: Path,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate Grad-CAM overlay image for a target class.
    Returns RGB uint8 image.
    """
    transform = get_preprocess_transform()
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((224, 224))

    rgb_img = np.array(resized_image).astype(np.float32) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Last convolutional block in ResNet18
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization


def save_gradcam_panel(
    original_image_path: Path,
    gradcam_overlay: np.ndarray,
    save_path: Path,
    true_label_value: int,
    pred_label_value: int,
    score_bad: float,
    bucket: str,
) -> None:
    """
    Saves a side-by-side panel:
    [original | grad-cam overlay]
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = Image.open(original_image_path).convert("RGB").resize((224, 224))
    original_np = np.array(original)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    fig.suptitle(
        f"{original_image_path.name} | "
        f"True: {LABEL_TO_CLASS[true_label_value]} ({true_label_value}) | "
        f"Pred: {LABEL_TO_CLASS[pred_label_value]} ({pred_label_value}) | "
        f"P(bad): {score_bad:.4f} | {bucket}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_path = Path(__file__).resolve().parent.parent

    prediction_images_root = base_path / "prediction_outputs_resnet18"
    test_dataset_root = base_path / "dataset" / "test"

    output_root = base_path / "evaluation_outputs"
    sorted_root = output_root / "sorted"
    gradcam_root = output_root / "gradcam"

    metrics_json = output_root / "summary_metrics.json"
    detailed_csv = output_root / "evaluated_predictions.csv"

    optional_scores_csv = prediction_images_root / "predictions.csv"

    # Update this path to your actual trained checkpoint
    checkpoint_path = base_path / "outputs" / "resnet18_binary" / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not prediction_images_root.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_images_root}\n"
            "Run inference first for the ResNet18 classifier."
        )

    if not test_dataset_root.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {test_dataset_root}")

    model = load_model(checkpoint_path, device)

    filename_to_score = load_optional_scores(optional_scores_csv)

    records = []
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for pred_folder in ["good", "bad"]:
        pred_path = prediction_images_root / pred_folder
        if not pred_path.exists():
            continue

        folder_pred_label = predicted_label_from_folder(pred_folder)

        for image_file in pred_path.iterdir():
            if not image_file.is_file():
                continue
            if image_file.suffix.lower() not in valid_extensions:
                continue

            filename = image_file.name
            original_path = None
            y_true = None

            for true_folder in ["good", "bad"]:
                candidate_path = test_dataset_root / true_folder / filename
                if candidate_path.exists():
                    original_path = candidate_path
                    y_true = class_label_from_name(true_folder)
                    break

            if original_path is None or y_true is None:
                print(f"Warning: Could not find original image for {filename}, skipping")
                continue

            # Re-run model on original image to get score
            # and optionally verify prediction
            model_pred_label, model_score_bad = predict_with_model(model, original_path, device)

            # Keep the prediction inferred from folder structure, since that is
            # what your inference pipeline exported.
            # But record the model prediction too for debugging.
            exported_pred_label = folder_pred_label

            score = filename_to_score.get(filename, model_score_bad)

            records.append(
                {
                    "filename": filename,
                    "image_path": str(image_file),
                    "original_image_path": str(original_path),
                    "pred_label": exported_pred_label,
                    "model_pred_label": model_pred_label,
                    "true_label": y_true,
                    "score": score,
                }
            )

    if not records:
        raise FileNotFoundError(
            f"No prediction images found in: {prediction_images_root}\n"
            "Make sure inference output is organized as:\n"
            "  prediction_outputs_resnet18/good/\n"
            "  prediction_outputs_resnet18/bad/"
        )

    df = pd.DataFrame(records)

    df["bucket"] = [
        bucket_name(y_true, y_pred)
        for y_true, y_pred in zip(df["true_label"], df["pred_label"])
    ]

    # Copy images and generate Grad-CAM
    for row in df.itertuples(index=False):
        image_path = Path(row.image_path)
        original_image_path = Path(row.original_image_path)

        if image_path.exists():
            destination = sorted_root / row.bucket / image_path.name
            safe_copy(image_path, destination)
        else:
            print(f"Warning: predicted output image not found, skipping copy: {image_path}")

        try:
            # Usually visualize the predicted class
            target_class = int(row.pred_label)

            overlay = generate_gradcam_overlay(
                model=model,
                image_path=original_image_path,
                target_class=target_class,
                device=device,
            )

            panel_path = gradcam_root / row.bucket / f"{Path(row.filename).stem}_gradcam.jpg"
            save_gradcam_panel(
                original_image_path=original_image_path,
                gradcam_overlay=overlay,
                save_path=panel_path,
                true_label_value=int(row.true_label),
                pred_label_value=int(row.pred_label),
                score_bad=float(row.score),
                bucket=row.bucket,
            )
        except Exception as e:
            print(f"Warning: failed to generate Grad-CAM for {row.filename}: {e}")

    y_true = df["true_label"].tolist()
    y_pred = df["pred_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

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

    has_scores = df["score"].notna().all()
    if has_scores and len(set(y_true)) == 2:
        y_score = df["score"].astype(float).tolist()
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None

    output_root.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good (0)", "Bad (1)"],
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("ResNet18 Binary Classification Confusion Matrix")
    plt.tight_layout()
    cm_image_path = output_root / "confusion_matrix.png"
    plt.savefig(cm_image_path, dpi=100, bbox_inches="tight")
    plt.close()

    cm_df = pd.DataFrame(
        cm,
        index=["True Good (0)", "True Bad (1)"],
        columns=["Pred Good (0)", "Pred Bad (1)"],
    )
    cm_csv_path = output_root / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    df.to_csv(detailed_csv, index=False)

    metrics_csv_path = output_root / "summary_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

    print("\nEvaluation complete.")
    print("\n" + "=" * 60)
    print("RESNET18 EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("=" * 60)

    print(f"\nDetailed predictions CSV: {detailed_csv}")
    print(f"Summary metrics JSON: {metrics_json}")
    print(f"Summary metrics CSV: {metrics_csv_path}")
    print(f"Confusion matrix CSV: {cm_csv_path}")
    print(f"Confusion matrix image: {cm_image_path}")
    print(f"Sorted images by bucket: {sorted_root}")
    print(f"Grad-CAM visualizations: {gradcam_root}")


if __name__ == "__main__":
    main()