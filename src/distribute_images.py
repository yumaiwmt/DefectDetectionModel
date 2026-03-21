import os
import random
import shutil
from pathlib import Path

# Define paths
base_path = Path(__file__).parent.parent
data_source_good = base_path / "data_source" / "good"
data_source_bad = base_path / "data_source" / "bad"

# Destination paths
train_good = base_path / "dataset" / "train" / "good"
val_good = base_path / "dataset" / "val" / "good"
val_bad = base_path / "dataset" / "val" / "bad"
test_good = base_path / "dataset" / "test" / "good"
test_bad = base_path / "dataset" / "test" / "bad"

# Create destination directories if they don't exist
for folder in [train_good, val_good, val_bad, test_good, test_bad]:
    folder.mkdir(parents=True, exist_ok=True)

# Get all good images and shuffle
good_images = os.listdir(data_source_good)
random.shuffle(good_images)

# Get all bad images and shuffle
bad_images = os.listdir(data_source_bad)
random.shuffle(bad_images)

# Distribution counts
good_distribution = {
    "train": (train_good, 600),
    "val": (val_good, 200),
    "test": (test_good, 200)
}

bad_distribution = {
    "val": (val_bad, 140),
    "test": (test_bad, 210)
}

# Distribute good images
print("Distributing good images...")
current_idx = 0
for split, (dest_folder, count) in good_distribution.items():
    for i in range(count):
        src = data_source_good / good_images[current_idx]
        dst = dest_folder / good_images[current_idx]
        shutil.copy2(src, dst)
        current_idx += 1
    print(f"  Copied {count} good images to {split}")

# Distribute bad images
print("Distributing bad images...")
current_idx = 0
for split, (dest_folder, count) in bad_distribution.items():
    for i in range(count):
        src = data_source_bad / bad_images[current_idx]
        dst = dest_folder / bad_images[current_idx]
        shutil.copy2(src, dst)
        current_idx += 1
    print(f"  Copied {count} bad images to {split}")

print("\nDistribution complete!")
print(f"  Total good images distributed: {sum(count for _, count in good_distribution.values())}")
print(f"  Total bad images distributed: {sum(count for _, count in bad_distribution.values())}")
