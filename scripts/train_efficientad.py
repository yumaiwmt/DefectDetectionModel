from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def main() -> None:
    # Paths
    base_path = Path(__file__).resolve().parent.parent
    dataset_root = base_path / "dataset"
    results_root = base_path / "outputs"

    train_good = dataset_root / "train" / "resized_aug_images"
    val_good = dataset_root / "val" / "good"
    val_bad = dataset_root / "val" / "bad"
    test_good = dataset_root / "test" / "good"
    test_bad = dataset_root / "test" / "bad"

    required_dirs = [train_good, val_good, val_bad, test_good, test_bad]
    for folder in required_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    # Datamodule
    datamodule = Folder(
        name="defect_dataset",
        root=dataset_root,
        normal_dir="train/resized_aug_images",
        abnormal_dir=["val/bad", "test/bad"],
        normal_test_dir=["val/good", "test/good"],
        train_batch_size=1,
        eval_batch_size=8,
        num_workers=0,
    )

    # Model
    model = EfficientAd()

    # Engine
    engine = Engine(
        default_root_dir=str(results_root),
        max_epochs=30,
        accelerator="auto",
        devices=1,
    )

    print("Starting EfficientAD training...")
    engine.fit(model=model, datamodule=datamodule)

    print("Running EfficientAD test evaluation...")
    test_results = engine.test(model=model, datamodule=datamodule)

    print("\nEfficientAD test results:")
    print(test_results)


if __name__ == "__main__":
    main()