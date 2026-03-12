from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main() -> None:
    # Paths
    base_path = Path(__file__).resolve().parent.parent
    dataset_root = base_path / "dataset"
    results_root = base_path / "outputs" / "patchcore"

    train_good = dataset_root / "train" / "resized_aug_images"
    val_good = dataset_root / "val" / "good"
    val_bad = dataset_root / "val" / "bad"
    test_good = dataset_root / "test" / "good"
    test_bad = dataset_root / "test" / "bad"

    required_dirs = [train_good, val_good, val_bad, test_good, test_bad]
    for folder in required_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    datamodule = Folder(
        name="defect_dataset",
        root=dataset_root,
        normal_dir="resized_aug_images",
        abnormal_dir="bad",
        normal_test_dir="good",
        task="classification",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore()

    engine = Engine(
        default_root_dir=str(results_root),
        max_epochs=1,
        accelerator="auto",
        devices=1,
    )

    datamodule.setup()
    print("Train images:", len(datamodule.train_data))
    print("Val images:", len(datamodule.val_data))
    print("Test images:", len(datamodule.test_data))

    print("Starting PatchCore training...")
    engine.fit(model=model, datamodule=datamodule)

    print("Running PatchCore test evaluation...")
    test_results = engine.test(model=model, datamodule=datamodule)

    print("\nPatchCore test results:")
    print(test_results)


if __name__ == "__main__":
    main()