# Defect Detection Model

A machine learning project for detecting defects in images. This repository supports two approaches:
- **Anomaly Detection** (main branch) - Using EfficientAd and PatchCore models
- **Binary Classification** (binary-classification branch) - Using ResNet18

## Installation

1. Clone the repository and navigate to the project directory
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- **Pillow** (>=10.0.0) - Image processing library
- **NumPy** (>=1.24.0) - Numerical computing library
- **Pandas** (>=1.5.0) - Data manipulation and analysis
- **Matplotlib** (>=3.7.0) - Visualization library
- **Anomalib** (>=1.0.0) - Anomaly detection library with pre-trained models
- **PyTorch Lightning** (>=2.0.0) - PyTorch training framework
- **PyTorch** (>=2.0.0) - Deep learning framework
- **TorchVision** (>=0.15.0) - Computer vision utilities for PyTorch

## Workflow

### Branch: `main` (Anomaly Detection)

This branch implements anomaly detection using unsupervised and semi-supervised learning. You can choose between two models:
- **EfficientAd** - Efficient and lightweight anomaly detection
- **PatchCore** - Feature-based anomaly detection with coreset sampling

#### Step 1: Prepare Data Source
Create a `data_source/` folder with good and bad images:

```
data_source/
├── good/    # Good/non-defective images
└── bad/     # Defective images
```

#### Step 2: Distribute Images
```bash
python src/distribute_images.py
```

This distributes images from `data_source/` into training, validation, and test sets:
- **Training**: `dataset/train/resized_aug_images/` (good images only)
- **Validation**: `dataset/val/good/` and `dataset/val/bad/`
- **Test**: `dataset/test/good/` and `dataset/test/bad/`

#### Step 3: Preprocess Data
```bash
python src/resize_transforms.py
```

Resizes and augments training images (384x384):
- Original resize
- Horizontal flip
- Vertical flip
- Micro affine transformation

#### Option 1: Using EfficientAd

**Step 4a: Train EfficientAd Model**
```bash
python scripts/train_efficientad.py
```

**Step 4b: Inference (EfficientAd)**
```bash
python scripts/inference_efficientad.py
```

**Step 4c: Evaluate (EfficientAd)**
```bash
python scripts/visualize_anomaly_map.py
```

---

#### Option 2: Using PatchCore

**Step 4a: Train PatchCore Model**
```bash
python scripts/train_patchcore.py
```

**Step 4b: Inference (PatchCore)**
```bash
python scripts/inference_patchcore.py
```

**Step 4c: Evaluate (PatchCore)**
```bash
python scripts/visualize_multiscale_maps.py
```

---

### Branch: `binary-classification` (Binary Classification)

This branch implements binary classification using ResNet18 to classify images as good or bad.

#### Step 1: Prepare Data Source
Create a `data_source/` folder with good and bad images (same as main branch):

```
data_source/
├── good/    # Good/non-defective images
└── bad/     # Defective images
```

#### Step 2: Distribute Images
**⚠️ Important**: Make sure the `dataset/` folder is empty before running this script.

```bash
python src/distribute_images.py
```

Distributes images into training and validation sets with binary labels.

#### Step 3: Train ResNet18 Model
```bash
python scripts/train_resnet18.py
```

Trains a binary classifier (good vs. bad). The model selection is based on validation F1 score. **Test set is never exposed during training.**

#### Step 4: Inference
```bash
python scripts/inference_resnet18.py
```

#### Step 5: Evaluate
```bash
python scripts/evaluate_resnet18.py
```

---

## Project Structure

```
DefectDetectionModel/
├── README.md
├── requirements.txt
├── data_source/                 # Raw source images (create this folder)
│   ├── good/                    
│   └── bad/                     
├── dataset/                     # Distributed training/validation images
│   ├── train/
│   ├── val/
│   └── test/
├── dataset_binary/              # Binary classification dataset (binary-classification branch)
├── models/                      # Model definitions
├── notebooks/                   # Jupyter notebooks
├── outputs/                     # Model checkpoints and results (anomaly detection)
├── pre_trained/                 # Pre-trained model weights
├── scripts/                     # Training and inference scripts
├── src/                         # Data preprocessing scripts
└── datasets/                    # External datasets (imagenette, etc.)
```

## Notes

- Always check that you're on the correct branch before running scripts
- For the binary-classification branch, ensure the `dataset/` folder is cleaned before distributing images
- PyTorch GPU acceleration is supported if CUDA is available

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

- URL to weight files: https://drive.google.com/drive/folders/1oDkp_GDH6tgAIwHAKAMyvkafx6fPIzyl?usp=drive_link
- "efficientad_model.ckpt" is the weight file for the EfficientAD model, originally named model.ckpt
- "patchcore_model.ckpt" is the weight file for the PatchCore model, originally named model.ckpt
- "resnet18_final_checkpoint.pt" is the checkpoint file for ResNet8