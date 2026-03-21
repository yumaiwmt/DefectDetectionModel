from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def main() -> None:
    torch.set_float32_matmul_precision("high")  # optional speedup on 3090

    base_path = Path(__file__).resolve().parent.parent
    dataset_root = base_path / "dataset"
    results_root = base_path / "outputs"

    train_good = dataset_root / "train" / "resized_aug_images"
    val_good = dataset_root / "val" / "good"
    val_bad = dataset_root / "val" / "bad"

    required_dirs = [train_good, val_good, val_bad]
    for folder in required_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    # optional sanity check
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for folder in required_dirs:
        count = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
        print(f"{folder}: {count} images")

    datamodule = Folder(
        name="defect_dataset",
        root=dataset_root,
        normal_dir="train/resized_aug_images",
        abnormal_dir="val/bad",
        normal_test_dir="val/good",
        train_batch_size=1,
        eval_batch_size=8,
        num_workers=0
    )

    model = EfficientAd()

    engine = Engine(
        default_root_dir=str(results_root),
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        precision="16-mixed"
    )

    print("Starting EfficientAD training...")
    engine.fit(model=model, datamodule=datamodule)

    print("Running EfficientAD test evaluation...")
    test_results = engine.test(model=model, datamodule=datamodule)

    print("\nEfficientAD test results:")
    print(test_results)


if __name__ == "__main__":
    main()