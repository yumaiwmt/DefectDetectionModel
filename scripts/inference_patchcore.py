from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    base_path = Path(__file__).resolve().parent.parent

    ckpt_path = base_path / "outputs_patchcore" / "Patchcore" / "defect_dataset" /"v2" / "weights" / "lightning" / "model.ckpt"
    images_to_test = base_path / "dataset" / "test"
    results_root = base_path / "prediction_outputs_patchcore"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not images_to_test.exists():
        raise FileNotFoundError(f"Images folder not found: {images_to_test}")

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
        num_neighbors=9,
    )
    
    predict_dataset = PredictDataset(path=images_to_test)

    engine = Engine(
        default_root_dir=str(results_root),
        accelerator="auto",
        devices=1,
    )

    engine.predict(
        model=model,
        dataset=predict_dataset,
        ckpt_path=str(ckpt_path),
    )


if __name__ == "__main__":
    main()