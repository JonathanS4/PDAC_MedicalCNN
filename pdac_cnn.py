from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    data_root: Path = Path(os.environ.get(
        "PDAC_DATA_ROOT",
        PROJECT_ROOT / "data" / "DATASET",
    ))
    train_subdir: str = "train/train"
    test_subdir: str = "test/test"
    class_names: tuple[str, ...] = ("normal", "pancreatic_tumor")
    val_fraction: float = 0.15

    image_size: int = 224
    backbone: str = "resnet18"          # resnet18 | resnet50
    freeze_backbone: bool = True        # feature-extraction: train head only
    dropout: float = 0.3
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    early_stopping_patience: int = 6
    target_accuracy: float = 0.95

    output_dir: Path = PROJECT_ROOT / "outputs"
    best_model_name: str = "best_model.pt"
    device: str = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        for d in (self.output_dir, self.checkpoint_dir, self.log_dir, self.figure_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def train_dir(self) -> Path:
        return self.data_root / self.train_subdir

    @property
    def test_dir(self) -> Path:
        return self.data_root / self.test_subdir

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / self.best_model_name


def build_transforms(cfg: Config):
    """Return (train_tf, eval_tf). CT slices are 1-channel; expand to 3 so we
    can reuse an ImageNet backbone. Augmentation is mild — aggressive geometric
    distortion is not anatomically meaningful for abdominal CT."""
    to_three_channel = transforms.Grayscale(num_output_channels=3)
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        to_three_channel,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        to_three_channel,
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])
    return train_tf, eval_tf


def _stratified_val_indices(targets, val_fraction, seed):
    """Split indices per class so both splits keep the class balance."""
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    targets_tensor = torch.tensor(targets)
    for class_id in targets_tensor.unique().tolist():
        positions = (targets_tensor == class_id).nonzero(as_tuple=True)[0]
        shuffled = positions[torch.randperm(len(positions), generator=generator)]
        n_val = int(round(len(shuffled) * val_fraction))
        val_idx.extend(shuffled[:n_val].tolist())
        train_idx.extend(shuffled[n_val:].tolist())
    return train_idx, val_idx


def _verify_class_order(dataset: ImageFolder, cfg: Config) -> None:
    if tuple(dataset.classes) != tuple(cfg.class_names):
        raise ValueError(
            f"Expected class folders {cfg.class_names} but found {dataset.classes} "
            f"in {dataset.root}. Fix folder names or Config.class_names so labels "
            "0/1 stay consistent between training and evaluation."
        )


def build_dataloaders(cfg: Config):
    """Return (train_loader, val_loader, test_loader). A stratified 15% val
    split is carved from train/ so the provided test/ set stays untouched."""
    train_tf, eval_tf = build_transforms(cfg)
    if not cfg.train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found at {cfg.train_dir}. Set PDAC_DATA_ROOT or "
            "pass --data-root."
        )

    full_train_aug = ImageFolder(str(cfg.train_dir), transform=train_tf)
    full_train_eval = ImageFolder(str(cfg.train_dir), transform=eval_tf)
    _verify_class_order(full_train_aug, cfg)

    train_idx, val_idx = _stratified_val_indices(
        full_train_aug.targets, cfg.val_fraction, cfg.seed)
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    test_ds = ImageFolder(str(cfg.test_dir), transform=eval_tf)
    _verify_class_order(test_ds, cfg)

    common = dict(num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_weights(cfg: Config) -> torch.Tensor:
    """Inverse-frequency class weights from the training folder for the loss."""
    counts = torch.zeros(len(cfg.class_names))
    dataset = ImageFolder(str(cfg.train_dir))
    _verify_class_order(dataset, cfg)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts.sum() / (len(counts) * counts)


def build_model(cfg: Config) -> nn.Module:
    """ImageNet-pretrained ResNet feature extractor + small classification head.
    On ~1,000 training slices this reaches the 95% target far more reliably than
    a from-scratch CNN, and — with the backbone frozen — trains fast on CPU."""
    backbone = cfg.backbone.lower()
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported backbone '{cfg.backbone}'. Use resnet18 or resnet50.")

    if cfg.freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(cfg.dropout),
        nn.Linear(in_features, len(cfg.class_names)),
    )
    for p in net.fc.parameters():  # head always trains, even if backbone frozen
        p.requires_grad = True
    return net.to(cfg.device)


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    labels: np.ndarray
    preds: np.ndarray
    probs: np.ndarray  # probability of positive class (pancreatic_tumor, idx 1)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_seen = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    torch.set_grad_enabled(is_train)
    for i, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        total_seen += n
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        if desc:
            print(f"\r  {desc}: batch {i}/{len(loader)}", end="", flush=True)
    if desc:
        print()
    torch.set_grad_enabled(True)

    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    probs_arr = np.concatenate(all_probs)
    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        accuracy=float((preds_arr == labels_arr).mean()),
        labels=labels_arr, preds=preds_arr, probs=probs_arr,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cmd_train(cfg: Config) -> None:
    set_seed(cfg.seed)
    print(f"Device: {cfg.device} | backbone: {cfg.backbone} | "
          f"frozen: {cfg.freeze_backbone} | epochs: {cfg.epochs}")
    print(f"Data root: {cfg.data_root}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss(weight=class_weights(cfg).to(cfg.device))
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    history, best_val_acc, no_improve, start = [], 0.0, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_res = run_epoch(model, train_loader, criterion, cfg.device,
                              optimizer=optimizer, desc=f"Epoch {epoch} [train]")
        val_res = run_epoch(model, val_loader, criterion, cfg.device,
                            desc=f"Epoch {epoch} [val]")
        scheduler.step(val_res.accuracy)
        history.append({"epoch": epoch, "train_loss": train_res.loss,
                        "train_acc": train_res.accuracy, "val_loss": val_res.loss,
                        "val_acc": val_res.accuracy,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:02d} | train loss {train_res.loss:.4f} "
              f"acc {train_res.accuracy:.4f} | val loss {val_res.loss:.4f} "
              f"acc {val_res.accuracy:.4f}")

        improved = val_res.accuracy > best_val_acc
        if improved:
            best_val_acc, no_improve = val_res.accuracy, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {"backbone": cfg.backbone, "image_size": cfg.image_size,
                           "dropout": cfg.dropout, "class_names": list(cfg.class_names),
                           "norm_mean": list(cfg.norm_mean), "norm_std": list(cfg.norm_std),
                           "freeze_backbone": cfg.freeze_backbone},
                "val_acc": best_val_acc, "epoch": epoch,
            }, cfg.best_model_path)
        else:
            no_improve += 1

        if best_val_acc >= cfg.target_accuracy and improved:
            print(f"Reached target validation accuracy "
                  f"({best_val_acc:.4f} >= {cfg.target_accuracy}).")
        if no_improve >= cfg.early_stopping_patience:
            print(f"Early stopping: no val improvement for "
                  f"{cfg.early_stopping_patience} epochs.")
            break

    (cfg.log_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining done in {(time.time() - start) / 60:.1f} min. "
          f"Best val acc: {best_val_acc:.4f}")
    print(f"Best model saved to {cfg.best_model_path}")
    print("Run `python pdac_cnn.py evaluate` for held-out test metrics.")


def load_model(cfg: Config) -> nn.Module:
    if not cfg.best_model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {cfg.best_model_path}. Train first: "
            "`python pdac_cnn.py train`.")
    ckpt = torch.load(cfg.best_model_path, map_location=cfg.device)
    saved = ckpt.get("config", {})
    cfg.backbone = saved.get("backbone", cfg.backbone)
    cfg.dropout = saved.get("dropout", cfg.dropout)
    cfg.freeze_backbone = saved.get("freeze_backbone", cfg.freeze_backbone)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')} "
          f"(val acc {ckpt.get('val_acc'):.4f}).")
    return model


def cmd_evaluate(cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless: write figures to disk
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                                 confusion_matrix, roc_auc_score, roc_curve)

    model = load_model(cfg)
    _, _, test_loader = build_dataloaders(cfg)
    res = run_epoch(model, test_loader, nn.CrossEntropyLoss(), cfg.device, desc="Test")

    report = classification_report(res.labels, res.preds,
                                   target_names=cfg.class_names, digits=4)
    auc = roc_auc_score(res.labels, res.probs)
    cm = confusion_matrix(res.labels, res.preds)

    print("\n===== Held-out test results =====")
    print(f"Accuracy : {res.accuracy:.4f}")
    print(f"ROC-AUC  : {auc:.4f}\n")
    print(report)
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix — held-out test set")
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(res.labels, res.probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — pancreatic_tumor vs normal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    metrics = {"accuracy": res.accuracy, "roc_auc": float(auc),
               "confusion_matrix": cm.tolist(), "target_accuracy": cfg.target_accuracy,
               "target_met": res.accuracy >= cfg.target_accuracy}
    (cfg.log_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    verdict = "MET" if metrics["target_met"] else "NOT met"
    print(f"\nTarget accuracy {cfg.target_accuracy:.0%}: {verdict} "
          f"(test accuracy {res.accuracy:.4f}).")
    print(f"Figures written to {cfg.figure_dir}/")


def cmd_predict(cfg: Config, image_path: str) -> None:
    from PIL import Image
    model = load_model(cfg)
    _, eval_tf = build_transforms(cfg)
    image = Image.open(image_path).convert("L")  # CT slices are grayscale
    tensor = eval_tf(image).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    tumor_prob = float(probs[1])
    label = cfg.class_names[int(probs.argmax())]
    print(f"\nImage      : {image_path}")
    print(f"Prediction : {label}")
    print(f"Tumor prob : {tumor_prob:.4f}  (normal prob {1 - tumor_prob:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pancreatic CT CNN classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch-size", type=int)
    p_train.add_argument("--lr", type=float)
    p_train.add_argument("--backbone", choices=["resnet18", "resnet50"])
    p_train.add_argument("--unfreeze", action="store_true",
                         help="Fine-tune the whole backbone (default: head only)")
    p_train.add_argument("--data-root")

    p_eval = sub.add_parser("evaluate", help="Evaluate on the held-out test set")
    p_eval.add_argument("--data-root")

    p_pred = sub.add_parser("predict", help="Classify a single CT slice")
    p_pred.add_argument("image", help="Path to a CT slice image (jpg/png)")
    p_pred.add_argument("--data-root")

    args = parser.parse_args()
    cfg = Config()
    if getattr(args, "data_root", None):
        cfg.data_root = Path(args.data_root)

    if args.command == "train":
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.learning_rate = args.lr
        if args.backbone is not None:
            cfg.backbone = args.backbone
        if args.unfreeze:
            cfg.freeze_backbone = False
        cmd_train(cfg)
    elif args.command == "evaluate":
        cmd_evaluate(cfg)
    elif args.command == "predict":
        cmd_predict(cfg, args.image)


if __name__ == "__main__":
    main()
