from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def run_inference(
    image_path_or_dir: str | Path,
    ckpt_path: str | Path,
    results_root: str | Path | None = None,
    image_size: tuple[int, int] = (256, 256),
):
    model = EfficientAd()

    engine = Engine(
        default_root_dir=str(results_root) if results_root is not None else None,
        accelerator="auto",
        devices=1,
    )

    dataset = PredictDataset(
        path=str(image_path_or_dir),
        image_size=image_size,
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
        return_predictions=True,
    )

    return predictions


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

    return predictions


if __name__ == "__main__":
    main()