from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference_efficientad import run_inference


def show_prediction(prediction):
    image = np.array(Image.open(prediction.image_path).convert("RGB"))
    anomaly_map = prediction.anomaly_map.cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map, cmap="jet")
    plt.title("Anomaly Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(anomaly_map, cmap="jet", alpha=0.45)
    plt.title(f"Score: {float(prediction.pred_score):.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    base_path = Path(__file__).resolve().parent.parent

    ckpt_path = base_path / "outputs" / "v1" / "weights" / "lightning" / "model.ckpt"
    images_to_test = base_path / "new_images"
    results_root = base_path / "prediction_outputs"

    predictions = run_inference(
        image_path_or_dir=images_to_test,
        ckpt_path=ckpt_path,
        results_root=results_root,
        image_size=(256, 256),
    )

    for batch in predictions:
        if isinstance(batch, list):
            for prediction in batch:
                show_prediction(prediction)
        else:
            show_prediction(batch)


if __name__ == "__main__":
    main()