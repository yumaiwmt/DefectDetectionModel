from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import random

# Paths

base_path = Path(__file__).parent.parent

good_input_folder = base_path / "dataset" / "train" / "good"
bad_input_folder = base_path / "dataset" / "train" / "bad"

output_root = base_path / "dataset_binary" / "train"
output_folder = output_root / "resized_aug_images"
labels_csv_path = output_root / "labels.csv"

output_folder.mkdir(parents=True, exist_ok=True)

# set global constants

TARGET_SIZE = (384, 384)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Number of augmented copies per image
N_AUG_GOOD = 1
N_AUG_BAD = 3

def resize_image(img: Image.Image) -> Image.Image:
    return img.resize(TARGET_SIZE, Image.Resampling.LANCZOS) 

def affine(img: Image.Image) -> Image.Image:
    """
    Apply a very small affine transform:
    - tiny translation
    - tiny scale change
    No strong rotation/shear here because small defects can be damaged easily.
    """

    width, height = img.size

    shift_x = random.randint(-max(1, int(width * 0.02)), max(1, int(width * 0.02)))
    shift_y = random.randint(-max(1, int(height * 0.02)), max(1, int(height * 0.02)))
    scale = random.uniform(0.98, 1.02)

    """
    PIL affine expects a 6-tuple:
    (a, b, c, d, e, f)
    x' = a*x + b*y + c
    y' = d*x + e*y + f
    
    a = x scaling, stretches/compresses horizontally
    b = x shear, slants the image horizontally
    c = x translation, moves the image left/right
    d = y shear, slants the image vertically
    e = y scaling, stretches/compresses vertically
    f = y translation, moves the image up/down
    """

    affine_matrix = (scale, 0, shift_x, 0, scale, shift_y)

    return img.transform(img.size, Image.AFFINE, affine_matrix, resample=Image.Resampling.BILINEAR)

def slight_rotation(img: Image.Image) -> Image.Image:
    """
    Apply a tiny rotation only.
    Keep this small because scratches/dents may be subtle.
    """
    angle = random.uniform(-3.0, 3.0)
    return img.rotate(
        angle,
        resample=Image.Resampling.BILINEAR
    )

def adjust_brightness(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.92, 1.08)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.92, 1.08)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def mild_blur(img: Image.Image) -> Image.Image:
    """
    Very light blur to simulate slight focus variation.
    Keep radius tiny so fine defects are not erased.
    """
    radius = random.uniform(0.2, 0.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def choose_augmented_version(img: Image.Image) -> Image.Image:
    """
    Build one mild augmented sample.
    We use small, realistic changes only.
    """
    aug_img = img.copy()

    # Small geometric variation with moderate probability
    if random.random() < 0.7:
        aug_img = affine(aug_img)

    # Very small rotation with moderate probability
    if random.random() < 0.4:
        aug_img = slight_rotation(aug_img)

    # Mild photometric variation
    if random.random() < 0.5:
        aug_img = adjust_brightness(aug_img)

    if random.random() < 0.5:
        aug_img = adjust_contrast(aug_img)

    # Light blur only occasionally
    if random.random() < 0.2:
        aug_img = mild_blur(aug_img)

    return aug_img

def get_image_files(folder: Path):
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

def process_folder(input_folder: Path, class_name: str, label: int, n_aug: int, records: list):
    image_files = get_image_files(input_folder)
    print(f"Found {len(image_files)} images in '{class_name}'.")

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                resized_img = resize_image(img)

                resized_name = f"{class_name}_{image_path.stem}_orig.png"
                resized_path = output_folder / resized_name
                resized_img.save(resized_path)

                records.append({
                    "filename": resized_name,
                    "label": label,
                    "class_name": class_name,
                    "source_file": str(image_path)
                })

                # Save augmented copies
                for i in range(n_aug):
                    aug_img = choose_augmented_version(resized_img)
                    aug_name = f"{class_name}_{image_path.stem}_aug_{i+1}.png"
                    aug_path = output_folder / aug_name
                    aug_img.save(aug_path)

                    records.append({
                        "filename": aug_name,
                        "label": label,
                        "class_name": class_name,
                        "source_file": str(image_path)
                    })

            print(f"Processed: {image_path.name}")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

# Main processing

records = []

process_folder(
    input_folder=good_input_folder,
    class_name="good",
    label=0,
    n_aug=N_AUG_GOOD,
    records=records
)

process_folder(
    input_folder=bad_input_folder,
    class_name="bad",
    label=1,
    n_aug=N_AUG_BAD,
    records=records
)

df = pd.DataFrame(records)
df.to_csv(labels_csv_path, index=False)

print(f"Saved labels CSV to: {labels_csv_path}")
print(f"Total output images: {len(df)}")
print("Done.")