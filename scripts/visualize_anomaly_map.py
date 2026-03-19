import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_prediction(prediction, threshold=0.5):
    image = np.array(Image.open(prediction.image_path).convert("RGB"))
    anomaly_map = prediction.anomaly_map.detach().cpu().numpy()

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.squeeze()

    binary_mask = anomaly_map > threshold

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(anomaly_map, cmap="jet")
    plt.title("Anomaly Map")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(image)
    plt.imshow(anomaly_map, cmap="jet", alpha=0.45)
    plt.title(f"Overlay | Score={float(prediction.pred_score):.3f}")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(binary_mask, cmap="gray")
    plt.title(f"Mask > {threshold}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()