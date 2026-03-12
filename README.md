# Defect Detection Model

A machine learning project for detecting defects in images using image processing and data augmentation techniques.

## Project Structure

```
DefectDetectionModel/
├── README.md
├── requirements.txt
├── data_source/                 # Raw source images
│   ├── good/                    # Good images (1000 images)
│   └── bad/                     # Defective images (350 images)
├── dataset/                     # Distributed and organized images
│   ├── train/
│   │   └── good/               # Training good images (700)
│   ├── val/
│   │   ├── good/               # Validation good images (150)
│   │   └── bad/                # Validation bad images (100)
│   └── test/
│       ├── good/               # Test good images (150)
│       └── bad/                # Test bad images (250)
├── processed_data/              # Augmented and transformed images
│   └── train/
│       ├── resized_images/      # Resized images (384x384)
│       ├── hflip/               # Horizontally flipped images
│       ├── vflip/               # Vertically flipped images
│       └── micro_affine/        # Affine transformed images
├── models/                      # Trained model files
├── notebooks/                   # Jupyter notebooks for experimentation
├── outputs/                     # Model outputs and predictions
└── src/                         # Source scripts
    ├── distribute_images.py     # Random image distribution script
    ├── resize_transforms.py     # Image resizing and augmentation script
    ├── split_data.py            # Data splitting script
    ├── train_patchcore.py       # PatchCore anomaly detection model training
    └── train_efficientad.py     # EfficientAD anomaly detection model training
```

## Installation

1. Clone the repository and navigate to the project directory
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- **Pillow** (>=10.0.0) - Image processing library
- **NumPy** (>=1.24.0) - Numerical computing library
- **Anomalib** (>=1.0.0) - Anomaly detection library with pre-trained models
- **PyTorch Lightning** (>=2.0.0) - PyTorch training framework
- **PyTorch** (>=2.0.0) - Deep learning framework
- **TorchVision** (>=0.15.0) - Computer vision utilities for PyTorch

## Data Setup

### Initial Data Structure (`data_source/`)

Before running any scripts, organize your raw images in the `data_source` folder:

```
data_source/
├── good/          # Good/non-defective images (1000 images)
└── bad/           # Defective images (350 images)
```

### Image Distribution (`dataset/`)

The `distribute_images.py` script randomly distributes images from `data_source/` into training, validation, and test sets according to the following distribution:

- **Training (700 good images)**: `dataset/train/good/`
- **Validation (150 good + 100 bad)**: `dataset/val/good/` and `dataset/val/bad/`
- **Testing (150 good + 250 bad)**: `dataset/test/good/` and `dataset/test/bad/`

## Scripts

### 1. distribute_images.py

Randomly distributes images from `data_source/` to the `dataset/` folder according to predefined ratios.

**Usage:**
```bash
python src/distribute_images.py
```

**Output:**
- Copies images to `/dataset/train/good/`, `/dataset/val/`, and `/dataset/test/`
- Prints distribution summary

### 2. resize_transforms.py

Processes training images by resizing and applying data augmentation techniques.

**Operations:**
- **Resizing**: Resizes all images to 384x384 pixels
- **Horizontal Flip**: Creates horizontally flipped versions
- **Vertical Flip**: Creates vertically flipped versions
- **Micro-Affine**: Applies small affine transformations (1% shift and scale)

**Usage:**
```bash
python src/resize_transforms.py
```

**Output:**
- Processed images saved to `/processed_data/train/` with subdirectories:
  - `resized_images/` - Resized images
  - `hflip/` - Horizontally flipped versions
  - `vflip/` - Vertically flipped versions
  - `micro_affine/` - Affine transformed versions

### 3. split_data.py

Utility script for data splitting (details to be implemented).

### 4. train_patchcore.py

Trains a PatchCore anomaly detection model for defect detection.

**Features:**
- Uses pre-trained CNN backbone for feature extraction
- Memory-efficient anomaly detection
- Suitable for industrial defect detection

**Usage:**
```bash
python src/train_patchcore.py
```

**Input:**
- Training images from `/dataset/train/resized_aug_images/`
- Validation good images from `/dataset/val/good/`
- Validation defects from `/dataset/val/bad/`
- Test images from `/dataset/test/good/` and `/dataset/test/bad/`

**Output:**
- Trained model and results saved to `/outputs/patchcore/`
- Training metrics and test results printed to console

### 5. train_efficientad.py

Trains an EfficientAD anomaly detection model for defect detection.

**Features:**
- Efficient anomaly detection with lower computational cost
- Pre-trained image encoder for feature extraction
- Suitable for lightweight deployment

**Usage:**
```bash
python src/train_efficientad.py
```

**Input:**
- Training images from `/dataset/train/resized_aug_images/`
- Validation good images from `/dataset/val/good/`
- Validation defects from `/dataset/val/bad/`
- Test images from `/dataset/test/good/` and `/dataset/test/bad/`

**Output:**
- Trained model and results saved to `/outputs/efficientad/`
- Training metrics and test results printed to console

## Workflow

1. **Prepare Source Data**: Place raw images in `data_source/good/` and `data_source/bad/`
2. **Distribute Data**: Run `distribute_images.py` to split data into train/val/test sets
3. **Process Images**: Run `resize_transforms.py` to resize and augment training images
4. **Model Training**: Choose and run one of the anomaly detection models:
   - `train_patchcore.py` - For memory-efficient anomaly detection
   - `train_efficientad.py` - For lightweight deployment
5. **Evaluation**: Model automatically evaluates on validation and test sets during training

## Model Comparison

| Model | Training Time | Memory Usage | Best For |
|-------|---------------|--------------|----------|
| **PatchCore** | Moderate | Moderate | Accurate anomaly detection |
| **EfficientAD** | Fast | Low | Lightweight deployment, edge devices |

## Notes

- All image distributions are randomized for reproducibility with different seeds
- Images are copied (not moved), preserving originals in `data_source/`
- Transformations are applied to training data only for augmentation purposes
- Supported image formats: PNG, JPG, JPEG, BMP, TIF, TIFF
- Training scripts expect processed images in `/dataset/train/resized_aug_images/`
- Model checkpoints and evaluation results are saved in `/outputs/` directory
