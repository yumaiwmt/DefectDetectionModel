from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def get_eval_transform(image_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    return model, class_names, image_size


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()


def main():
    base_dir = Path(__file__).resolve().parent.parent

    checkpoint_path = base_dir / "outputs" / "resnet18_binary" / "final_checkpoint.pt"
    image_dir = base_dir / "dataset" / "test"
    output_dir = base_dir / "prediction_outputs_resnet18"

    good_dir = output_dir / "good"
    bad_dir = output_dir / "bad"

    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, class_names, image_size = load_model(checkpoint_path, device)
    transform = get_eval_transform(image_size)

    print("Class names:", class_names)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    image_paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
        ]
        
    print(f"Found {len(image_paths)} images")
    
    for image_path in image_paths:

        pred, conf = predict_image(model, image_path, transform, device)
        class_name = class_names[pred]

        print(f"{image_path.name} -> {class_name} ({conf:.3f})")

        # Copy image to result folder
        if class_name == "good":
            shutil.copy(image_path, good_dir / image_path.name)
        else:
            shutil.copy(image_path, bad_dir / image_path.name)


if __name__ == "__main__":
    main()