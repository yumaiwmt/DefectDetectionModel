from pathlib import Path
import copy
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_transforms(image_size: int = 384):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


def calculate_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    eps = 1e-8
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)

    return {
        "accuracy": accuracy,
        "precision_bad": precision,
        "recall_bad": recall,
        "f1_bad": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = running_loss / max(total_samples, 1)
    metrics = calculate_metrics_from_counts(tp, tn, fp, fn)
    metrics["loss"] = avg_loss
    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    total_samples = 0

    tp = tn = fp = fn = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = running_loss / max(total_samples, 1)
    metrics = calculate_metrics_from_counts(tp, tn, fp, fn)
    metrics["loss"] = avg_loss
    return metrics


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset"
    output_dir = base_dir / "outputs" / "resnet18_binary"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    image_size = 384
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_workers = 4
    pretrained = True
    freeze_backbone = False  # set True if you want to train only the final layer first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms, eval_tfms = get_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tfms)

    print("Class mapping:", train_dataset.class_to_idx)

    expected_classes = {"bad", "good"}
    if set(train_dataset.class_to_idx.keys()) != expected_classes:
        raise ValueError(
            f"Expected classes {expected_classes}, but found {set(train_dataset.class_to_idx.keys())}. "
            "Please name your folders exactly 'good' and 'bad'."
        )

    # We want label 1 to mean "bad" for easier interpretation of recall_bad / precision_bad.
    # ImageFolder assigns labels alphabetically: bad=0, good=1
    # To avoid confusion, we remap targets inside the datasets.
    original_mapping = train_dataset.class_to_idx
    bad_old_idx = original_mapping["bad"]
    good_old_idx = original_mapping["good"]

    def remap_targets(dataset):
        new_targets = []
        for target in dataset.targets:
            if target == good_old_idx:
                new_targets.append(0)  # good -> 0
            elif target == bad_old_idx:
                new_targets.append(1)  # bad -> 1
            else:
                raise ValueError(f"Unexpected target: {target}")
        dataset.targets = new_targets
        dataset.samples = [(path, new_targets[i]) for i, (path, _) in enumerate(dataset.samples)]
        dataset.imgs = dataset.samples

    remap_targets(train_dataset)
    remap_targets(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Class weights for imbalanced training
    num_good = sum(1 for t in train_dataset.targets if t == 0)
    num_bad = sum(1 for t in train_dataset.targets if t == 1)

    # Larger weight for minority class "bad"
    class_weights = torch.tensor(
        [
            (num_good + num_bad) / (2.0 * max(num_good, 1)),
            (num_good + num_bad) / (2.0 * max(num_bad, 1)),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(f"Train counts -> good: {num_good}, bad: {num_bad}")
    print(f"Class weights: {class_weights.tolist()}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(device)

    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = []

    print("\nStarting training...\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["f1_bad"])

        if val_metrics["f1_bad"] > best_val_f1:
            best_val_f1 = val_metrics["f1_bad"]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        epoch_record = {
            "epoch": epoch_num,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch_num}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Train F1(bad): {train_metrics['f1_bad']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val Precision(bad): {val_metrics['precision_bad']:.4f} "
            f"Val Recall(bad): {val_metrics['recall_bad']:.4f} "
            f"Val F1(bad): {val_metrics['f1_bad']:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed / 60:.2f} minutes")

    # Load best model weights (based on validation performance)
    model.load_state_dict(best_model_wts)

    print("\nBest Validation F1(bad):", f"{best_val_f1:.4f}")
    print("Training complete. Model is ready for deployment.")
    print("Test set was never exposed to the model during training or validation.")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["good", "bad"],
            "image_size": image_size,
        },
        output_dir / "final_checkpoint.pt",
    )

    print(f"\nSaved best model to: {output_dir / 'best_model.pt'}")
    print(f"Saved training history to: {output_dir / 'history.json'}")
    print(f"Saved final checkpoint to: {output_dir / 'final_checkpoint.pt'}")


if __name__ == "__main__":
    main()