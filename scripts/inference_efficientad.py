from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def main():
    base_path = Path(__file__).resolve().parent.parent

    ckpt_path = base_path / "outputs" / "v1" / "weights" / "lightning" / "model.ckpt"
    images_to_test = base_path / "new_images"
    results_root = base_path / "prediction_outputs"

    model = EfficientAd()

    engine = Engine(
        default_root_dir=str(results_root),
        accelerator="auto",
        devices=1,
    )

    engine.predict(
        model=model,
        dataset=images_to_test,
        ckpt_path=str(ckpt_path),
    )


if __name__ == "__main__":
    main()