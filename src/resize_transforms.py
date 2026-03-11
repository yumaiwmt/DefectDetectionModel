from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

# Paths

base_path = Path(__file__).parent.parent

input_folder = base_path / "dataset" / "train" / "good"

output_base = base_path / "processed_data" / "train"
resized_folder = output_base / "resized_images"
hflip_folder = output_base / "hflip"
vflip_folder = output_base / "vflip"
affine_folder = output_base / "micro_affine"

output_folders = [
    resized_folder,
    hflip_folder,
    vflip_folder,
    affine_folder
]

for folder in output_folders:
    folder.mkdir(parents=True, exist_ok=True)

# set global constants

TARGET_SIZE = (384, 384)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def resize_image(img: Image.Image) -> Image.Image:
    return img.resize(TARGET_SIZE)


def horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def micro_affine(img: Image.Image) -> Image.Image:
    """
    Apply a small affine transform:
    - 1% shift in x and y
    - 1% scale
    """
    width, height = img.size
    shift_x = int(width * 0.01)
    shift_y = int(height * 0.01)
    scale = 1.01

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

    return img.transform(img.size, Image.AFFINE, affine_matrix)

# Main processing

image_files = [
    f for f in input_folder.iterdir()
    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
]

print(f"Found {len(image_files)} training images.")

for image_path in image_files:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # resize
            resized_img = resize_image(img)
            resized_output_path = resized_folder / image_path.name
            resized_img.save(resized_output_path)

            # horizontal flip (based on resized image)
            hflip_img = horizontal_flip(resized_img)
            hflip_output_path = hflip_folder / f"{image_path.stem}_hflip{image_path.suffix}"
            hflip_img.save(hflip_output_path)

            # vertical flip (based on resized image)
            vflip_img = vertical_flip(resized_img)
            vflip_output_path = vflip_folder / f"{image_path.stem}_vflip{image_path.suffix}"
            vflip_img.save(vflip_output_path)

            # micro-affine (based on resized image)
            affine_img = micro_affine(resized_img)
            affine_output_path = affine_folder / f"{image_path.stem}_affine{image_path.suffix}"
            affine_img.save(affine_output_path)

        print(f"Processed: {image_path.name}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

print("Done.")