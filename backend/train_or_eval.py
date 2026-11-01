#!/usr/bin/env python3
"""
Training & Evaluation Pipeline for Personalized CVD Restormer.
Directly aligned with: a-lightweight-conditional-restormer-for-multi-type (1).ipynb
Supports: Restormer_RDB (Ablation peak / Default Production), Modified_Conditional_Restormer,
Base_Restormer_Style, Restormer_CBAM, Plain_Conditional_Residual_CNN, Conditional_Residual_UNet.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TF
from torchvision.transforms import InterpolationMode
from pytorch_msssim import ms_ssim, ssim
import kornia
from tqdm import tqdm

from app.core.model import create_model, count_trainable_parameters, ALL_MODEL_NAMES
from app.config import settings

CVD_TYPES = ("Protanopia", "Deuteranopia", "Tritanopia")
CONDITION_TO_ID = {name: i for i, name in enumerate(CVD_TYPES)}
ID_TO_CONDITION = {i: name for name, i in CONDITION_TO_ID.items()}

# CVD simulation matrices from the notebook
CVD_MATRICES = torch.tensor(
    [
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ],
    ],
    dtype=torch.float32,
)


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.clamp(min=0).pow(1 / 2.4) - 0.055,
    )


def simulate_cvd(image: torch.Tensor, condition_id: torch.Tensor) -> torch.Tensor:
    matrices = CVD_MATRICES.to(image.device)[condition_id]
    linear = srgb_to_linear(image.clamp(0.0, 1.0))
    transformed = torch.einsum("bij,bjhw->bihw", matrices, linear)
    return linear_to_srgb(transformed).clamp(0.0, 1.0)


def normalized_lab(image: torch.Tensor) -> torch.Tensor:
    lab = kornia.color.rgb_to_lab(image.clamp(0.0, 1.0))
    lightness = lab[:, 0:1] / 100.0
    chroma = lab[:, 1:3] / 128.0
    return torch.cat([lightness, chroma], dim=1)


class CommonColourAwareLoss(nn.Module):
    """
    Multi-component Color-Aware Loss function from notebook.
    Combines: L1 loss + MS-SSIM loss + Normalized CIELAB loss + (optional) Simulation consistency.
    """
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ms_ssim: float = 0.30,
        lambda_lab: float = 0.15,
        lambda_simulation: float = 0.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ms_ssim = lambda_ms_ssim
        self.lambda_lab = lambda_lab
        self.lambda_simulation = lambda_simulation

    def forward(
        self,
        original: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        condition_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(prediction, target)

        structural = 1.0 - ms_ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True,
            win_size=7,
        )

        lab_loss = F.smooth_l1_loss(
            normalized_lab(prediction),
            normalized_lab(target),
        )

        if self.lambda_simulation > 0:
            simulated_prediction = simulate_cvd(prediction, condition_id)
            simulation_loss = F.l1_loss(simulated_prediction, original)
        else:
            simulation_loss = prediction.new_tensor(0.0)

        total = (
            self.lambda_l1 * l1
            + self.lambda_ms_ssim * structural
            + self.lambda_lab * lab_loss
            + self.lambda_simulation * simulation_loss
        )

        return {
            "total": total,
            "l1": l1.detach(),
            "ms_ssim_loss": structural.detach(),
            "lab": lab_loss.detach(),
            "simulation": simulation_loss.detach(),
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairedCVDDataset(Dataset):
    """
    Paired dataset matching notebook's PairedTransform pipeline:
    - Training: 1.12x resize -> random crop -> random horizontal flip -> random rotation [-5, 5] -> tensor
    - Validation/Testing: direct resize -> tensor
    """
    def __init__(self, pairs: List[Dict], image_size: int = 128, training: bool = True):
        self.pairs = pairs
        self.image_size = image_size
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        original_img = Image.open(item["original_path"]).convert("RGB")
        target_img = Image.open(item["target_path"]).convert("RGB")

        if self.training:
            resize_size = int(self.image_size * 1.12)
            original_img = TF.functional.resize(
                original_img, [resize_size, resize_size], interpolation=InterpolationMode.BICUBIC
            )
            target_img = TF.functional.resize(
                target_img, [resize_size, resize_size], interpolation=InterpolationMode.BICUBIC
            )

            max_offset = resize_size - self.image_size
            top = random.randint(0, max_offset)
            left = random.randint(0, max_offset)

            original_img = TF.functional.crop(original_img, top, left, self.image_size, self.image_size)
            target_img = TF.functional.crop(target_img, top, left, self.image_size, self.image_size)

            if random.random() < 0.5:
                original_img = TF.functional.hflip(original_img)
                target_img = TF.functional.hflip(target_img)

            angle = random.uniform(-5.0, 5.0)
            original_img = TF.functional.rotate(
                original_img, angle, interpolation=InterpolationMode.BILINEAR
            )
            target_img = TF.functional.rotate(
                target_img, angle, interpolation=InterpolationMode.BILINEAR
            )
        else:
            original_img = TF.functional.resize(
                original_img, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC
            )
            target_img = TF.functional.resize(
                target_img, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC
            )

        orig_tensor = TF.functional.to_tensor(original_img)
        targ_tensor = TF.functional.to_tensor(target_img)

        return {
            "image": orig_tensor,
            "target": targ_tensor,
            "condition_id": torch.tensor(item["condition_id"], dtype=torch.long),
            "condition": item["condition"],
            "image_id": item["image_id"],
        }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Computes PSNR, SSIM, MS-SSIM, MAE, MSE, and CIELAB Delta E evaluation metrics."""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True).item()
    ms_ssim_val = ms_ssim(pred, target, data_range=1.0, size_average=True, win_size=7).item()

    # CIELAB Delta E
    with torch.no_grad():
        lab_pred = kornia.color.rgb_to_lab(pred.clamp(0.0, 1.0))
        lab_target = kornia.color.rgb_to_lab(target.clamp(0.0, 1.0))
        delta_e = torch.sqrt(torch.sum((lab_pred - lab_target) ** 2, dim=1)).mean().item()

    return {
        "psnr": psnr,
        "ssim": ssim_val,
        "ms_ssim": ms_ssim_val,
        "mae": mae,
        "mse": mse,
        "delta_e": delta_e,
    }


def build_dataset_pairs(
    data_root: Path,
    seed: int = 2026,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Deterministic identity-grouped splitting (70% train / 15% val / 15% test).
    Guarantees no data leakage between splits.
    """
    original_dir = data_root / "original"
    corrected_dir = data_root / "corrected"

    if not original_dir.exists() or not corrected_dir.exists():
        raise FileNotFoundError(f"Missing original or corrected directories in {data_root}")

    originals = {
        f.stem.strip().lower(): f
        for f in original_dir.glob("*.*")
        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
    }
    all_pairs = []

    for condition in CVD_TYPES:
        cond_dir = corrected_dir / condition.lower()
        if not cond_dir.exists():
            continue
        targets = {
            f.stem.strip().lower(): f
            for f in cond_dir.glob("*.*")
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        }

        for image_id, orig_path in originals.items():
            if image_id in targets:
                all_pairs.append({
                    "image_id": image_id,
                    "condition": condition,
                    "condition_id": CONDITION_TO_ID[condition],
                    "original_path": orig_path,
                    "target_path": targets[image_id],
                })

    # Group by identity to prevent data leakage (matching notebook protocol)
    identity_list = sorted(list(originals.keys()))
    rng = np.random.default_rng(seed)
    shuffled_identities = list(rng.permutation(identity_list))

    n_total = len(shuffled_identities)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))

    train_ids = set(shuffled_identities[:n_train])
    val_ids = set(shuffled_identities[n_train:n_train + n_val])
    test_ids = set(shuffled_identities[n_train + n_val:])

    train_pairs = [p for p in all_pairs if p["image_id"] in train_ids]
    val_pairs = [p for p in all_pairs if p["image_id"] in val_ids]
    test_pairs = [p for p in all_pairs if p["image_id"] in test_ids]

    print(f"Identities ({n_total}) -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Dataset pairs -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


def find_default_data_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [
        repo_root / "dataset",
        repo_root / "archive" / "data",
        repo_root / "data",
        script_dir / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "archive" / "data",
    ]
    for candidate in candidates:
        if (candidate / "original").exists() and (candidate / "corrected").exists():
            return candidate.resolve()
    return (repo_root / "dataset").resolve()


def evaluate_model(
    model: nn.Module,
    pairs: List[Dict],
    batch_size: int,
    device: torch.device,
    num_workers: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Evaluates on dataset pairs with per-condition and overall breakdown."""
    test_loader = DataLoader(
        PairedCVDDataset(pairs, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()

    condition_metrics = {c: [] for c in CVD_TYPES}
    all_metrics = []
    latencies = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            cond_ids = batch["condition_id"].to(device)
            cond_names = batch["condition"]

            t0 = time.perf_counter()
            preds = model(imgs, cond_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000 / imgs.size(0))

            for i in range(imgs.size(0)):
                p = preds[i:i+1]
                t = targets[i:i+1]
                m = compute_metrics(p, t)
                condition_metrics[cond_names[i]].append(m)
                all_metrics.append(m)

    results = {}
    results["Overall"] = {
        "psnr": float(np.mean([m["psnr"] for m in all_metrics])),
        "ssim": float(np.mean([m["ssim"] for m in all_metrics])),
        "ms_ssim": float(np.mean([m["ms_ssim"] for m in all_metrics])),
        "mae": float(np.mean([m["mae"] for m in all_metrics])),
        "mse": float(np.mean([m["mse"] for m in all_metrics])),
        "delta_e": float(np.mean([m["delta_e"] for m in all_metrics])),
        "latency_per_image_ms": float(np.mean(latencies)),
    }

    for c in CVD_TYPES:
        c_list = condition_metrics[c]
        if c_list:
            results[c] = {
                "psnr": float(np.mean([m["psnr"] for m in c_list])),
                "ssim": float(np.mean([m["ssim"] for m in c_list])),
                "ms_ssim": float(np.mean([m["ms_ssim"] for m in c_list])),
                "mae": float(np.mean([m["mae"] for m in c_list])),
                "mse": float(np.mean([m["mse"] for m in c_list])),
                "delta_e": float(np.mean([m["delta_e"] for m in c_list])),
                "count": len(c_list),
            }

    return results


def main():
    default_data_dir = find_default_data_root()
    default_weights_path = (Path(__file__).resolve().parent / "weights" / "best_restormer.pt").resolve()

    parser = argparse.ArgumentParser(description="Train or evaluate CVD Restoration models from notebook.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Restormer_RDB",
        choices=list(ALL_MODEL_NAMES),
        help="Model architecture from notebook (default: Restormer_RDB)",
    )
    parser.add_argument("--data-root", type=str, default=str(default_data_dir), help="Path to dataset root")
    parser.add_argument("--weights-out", type=str, default=str(default_weights_path), help="Path to save best weights")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to evaluate or inspect")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate on test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs (default: 10)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional subset of training samples for quick runs")
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint; random weights are not a valid evaluation")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | Model architecture: {args.model_name}")

    data_root = Path(args.data_root)
    if not data_root.exists() or not (data_root / "original").exists():
        repo_alt = Path(__file__).resolve().parent.parent / args.data_root
        if repo_alt.exists() and (repo_alt / "original").exists():
            data_root = repo_alt

    train_pairs, val_pairs, test_pairs = build_dataset_pairs(data_root, seed=2026)

    # Initialize model
    model = create_model(model_name=args.model_name).to(device)
    total_params = count_trainable_parameters(model)
    print(f"Model: {args.model_name} | Trainable parameters: {total_params:,}")

    # Load checkpoint if specified for eval
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items() if "total_ops" not in k and "total_params" not in k}
        model.load_state_dict(clean_sd)
        print("Checkpoint loaded successfully.")

    if args.eval_only:
        print("\n--- Running Evaluation on Held-Out Test Set (128 identities, 384 pairs) ---")
        eval_results = evaluate_model(model, test_pairs, args.batch_size, device, num_workers=args.num_workers)
        print(f"\n=======================================================")
        print(f"EVALUATION RESULTS FOR {args.model_name}")
        print(f"=======================================================")
        for section, metrics in eval_results.items():
            print(f"\n[{section}]:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k:22s}: {v:.4f}")
                else:
                    print(f"  {k:22s}: {v}")
        return

    # Training mode
    if args.max_train_samples:
        train_pairs = train_pairs[:args.max_train_samples]
        print(f"Limiting training to {len(train_pairs)} samples.")

    train_loader = DataLoader(
        PairedCVDDataset(train_pairs, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PairedCVDDataset(val_pairs, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = CommonColourAwareLoss(lambda_l1=1.0, lambda_ms_ssim=0.30, lambda_lab=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_loss = float("inf")

    # Resume if requested
    if args.resume and Path(args.resume).exists():
        print(f"Resuming training from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items() if "total_ops" not in k and "total_params" not in k}
        model.load_state_dict(clean_sd)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_loss" in ckpt:
            best_loss = ckpt["val_loss"]
        print(f"Resumed at epoch {start_epoch}, best previous val loss: {best_loss:.5f}")

    weights_out = Path(args.weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = weights_out.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = weights_out.parent / "last_checkpoint.pt"
    audit_file = (Path(__file__).resolve().parent.parent / "audit.txt").resolve()
    no_improve_epochs = 0

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            cond_ids = batch["condition_id"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(imgs, cond_ids)
                loss_dict = criterion(imgs, preds, targets, cond_ids)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_pairs)

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                targets = batch["target"].to(device)
                cond_ids = batch["condition_id"].to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    preds = model(imgs, cond_ids)
                    loss_dict = criterion(imgs, preds, targets, cond_ids)
                    loss = loss_dict["total"]
                val_loss += loss.item() * imgs.size(0)
                val_metrics.append(compute_metrics(preds, targets))

        val_loss /= len(val_pairs)
        scheduler.step(val_loss)

        mean_psnr = np.mean([m["psnr"] for m in val_metrics])
        mean_ms_ssim = np.mean([m["ms_ssim"] for m in val_metrics])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {mean_psnr:.2f} dB | Val MS-SSIM: {mean_ms_ssim:.4f} | LR: {current_lr:.2e} | "
            f"Duration: {epoch_duration:.1f}s"
        )

        checkpoint_data = {
            "architecture": args.model_name,
            "model_version": "1.0.0",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_metric": min(best_loss, val_loss),
            "val_loss": val_loss,
            "val_psnr": float(mean_psnr),
            "val_ms_ssim": float(mean_ms_ssim),
            "train_loss": float(train_loss),
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "image_size": 128,
                "train_frac": 0.70,
                "val_frac": 0.15,
                "test_frac": 0.15,
            },
            "condition_mapping": {"Protanopia": 0, "Deuteranopia": 1, "Tritanopia": 2},
            "preprocessing_config": {
                "normalization": "[0.0, 1.0]",
                "crop_size": 128,
                "resize_ratio": 1.12,
            },
        }

        # Save last checkpoint
        torch.save(checkpoint_data, last_checkpoint_path)

        # Save epoch checkpoint in checkpoints/
        epoch_path = checkpoints_dir / f"{args.model_name}_epoch_{epoch:03d}.pt"
        torch.save(checkpoint_data, epoch_path)

        # If new best, save to best_restormer.pt (weights_out)
        is_new_best = val_loss < best_loss
        if is_new_best:
            best_loss = val_loss
            no_improve_epochs = 0
            torch.save(checkpoint_data, weights_out)
            print(f"  --> Saved new best checkpoint to: {weights_out}")
        else:
            no_improve_epochs += 1

        # Append structured progress entry to audit.txt
        if audit_file.exists():
            with open(audit_file, "a", encoding="utf-8") as af:
                af.write(
                    f"* Epoch {epoch:02d}:\n"
                    f"  - Train Loss: {train_loss:.5f}\n"
                    f"  - Val Loss:   {val_loss:.5f}\n"
                    f"  - Val PSNR:   {mean_psnr:.2f} dB\n"
                    f"  - Val MS-SSIM: {mean_ms_ssim:.4f}\n"
                    f"  - Learning Rate: {current_lr:.2e}\n"
                    f"  - Epoch Duration: {epoch_duration:.1f}s\n"
                    f"  - New Best Checkpoint: {'YES' if is_new_best else 'No'}\n"
                )

        if no_improve_epochs >= args.patience:
            print(f"\n[EarlyStopping] No validation loss improvement for {args.patience} epochs. Stopping training early at epoch {epoch:02d}.")
            break

    print("\nTraining completed successfully!")

    # Evaluate best checkpoint on test set
    if weights_out.exists():
        print("\n--- Evaluating Best Checkpoint on Test Set ---")
        best_ckpt = torch.load(weights_out, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_results = evaluate_model(model, test_pairs, args.batch_size, device, num_workers=args.num_workers)
        print(f"\nTest Set Results for {args.model_name}:")
        for s, m in test_results.items():
            print(f"  [{s}] PSNR: {m['psnr']:.4f} dB | SSIM: {m['ssim']:.4f} | MS-SSIM: {m['ms_ssim']:.4f} | MAE: {m['mae']:.4f} | Delta E: {m['delta_e']:.4f}")


if __name__ == "__main__":
    main()
