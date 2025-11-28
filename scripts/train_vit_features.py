#!/usr/bin/env python
"""
Training script for ViT-based feature extractor.

Self-supervised training using homography-based correspondences from HPatches dataset.

Usage:
    python scripts/train_vit_features.py --data-root data/raw/HPatches --epochs 100
    python scripts/train_vit_features.py --data-root data/raw/HPatches --resume checkpoints/latest.pt
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Suppress duplicate library warnings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_logging(log_dir: Path, experiment_name: str):
    """Setup logging directory and files."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_file = log_dir / f"{experiment_name}.log"

    return log_file


def log_message(message: str, log_file: Path = None, print_msg: bool = True):
    """Log message to console and optionally to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"

    if print_msg:
        print(formatted_msg)

    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(formatted_msg + "\n")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    losses,
    checkpoint_path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "losses": losses,
    }
    torch.save(checkpoint, checkpoint_path)


def diagnose_scheduler(scheduler, epoch, log_file):
    """Log detailed scheduler state for debugging."""
    log_message("Scheduler Diagnostics:", log_file)
    log_message(f"  Type: {type(scheduler).__name__}", log_file)
    log_message(f"  last_epoch: {scheduler.last_epoch}", log_file)
    log_message(f"  Current LR: {scheduler.get_last_lr()}", log_file)
    log_message(f"  Base LRs: {scheduler.base_lrs}", log_file)
    if hasattr(scheduler, "T_max"):
        log_message(f"  T_max: {scheduler.T_max}", log_file)
    if hasattr(scheduler, "eta_min"):
        log_message(f"  eta_min: {scheduler.eta_min}", log_file)


def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✓ Optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if checkpoint["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"✓ Scheduler state loaded (last_epoch: {scheduler.last_epoch})")
            except Exception as e:
                print(f"⚠️  Warning: Could not load scheduler state: {e}")
                print("    Scheduler will be manually synced")
        else:
            print("⚠️  Warning: No scheduler state in checkpoint")

    return checkpoint["epoch"], checkpoint["step"], checkpoint.get("losses", {})


def train_one_epoch(
    model,
    dataloader,
    processor,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epoch,
    log_interval,
    log_file,
    scaler=None,
    use_amp=False,
):
    """Train for one epoch."""
    model.train()

    epoch_losses = {
        "total": 0.0,
        "detector": 0.0,
        "detector_score": 0.0,
        "detector_orient": 0.0,
        "descriptor": 0.0,
    }
    num_batches = 0

    start_time = time.time()

    # Create progress bar
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}",
        unit="batch",
        leave=True,
    )

    for batch_idx, batch in pbar:
        # Move data to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        try:
            # Use automatic mixed precision if enabled
            with autocast(device_type=device.type, enabled=use_amp):
                # Process batch to get positive/negative samples
                outputs, targets = processor.process_batch(batch)

                # Compute losses
                losses = loss_fn(outputs, targets)

            # Backward pass with gradient scaling for AMP
            optimizer.zero_grad()

            if scaler is not None:
                scaler.scale(losses["total"]).backward()

                # Gradient clipping (unscale first for proper clipping)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # Update scheduler if step-based
            if scheduler is not None and hasattr(scheduler, "step_batch"):
                scheduler.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "det": f"{losses['detector'].item():.4f}",
                    "desc": f"{losses['descriptor'].item():.4f}",
                }
            )

            # Log to file at intervals (less verbose than before since tqdm shows progress)
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed

                log_message(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Speed: {batches_per_sec:.2f} batch/s",
                    log_file,
                    print_msg=False,  # Don't print since tqdm shows progress
                )

        except Exception as e:
            log_message(f"Error in batch {batch_idx}: {e}", log_file)
            import traceback

            traceback.print_exc()
            continue

    pbar.close()

    # Average losses
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

    epoch_time = time.time() - start_time

    log_message(
        f"Epoch {epoch} completed in {epoch_time:.1f}s | "
        f"Avg Loss: {epoch_losses['total']:.4f} | "
        f"Det: {epoch_losses['detector']:.4f} "
        f"(Score: {epoch_losses['detector_score']:.4f}, Orient: {epoch_losses['detector_orient']:.4f}) | "
        f"Desc: {epoch_losses['descriptor']:.4f}",
        log_file,
    )

    return epoch_losses


def validate(model, dataloader, processor, loss_fn, device, log_file, use_amp=False):
    """Run validation."""
    model.eval()

    val_losses = {
        "total": 0.0,
        "detector": 0.0,
        "detector_score": 0.0,
        "detector_orient": 0.0,
        "descriptor": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation", unit="batch", leave=False)

    with torch.no_grad():
        for batch in pbar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs, targets = processor.process_batch(batch)
                    losses = loss_fn(outputs, targets)

                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})
            except Exception as e:
                log_message(f"Validation error: {e}", log_file)
                continue

    pbar.close()

    if num_batches > 0:
        for key in val_losses:
            val_losses[key] /= num_batches

    log_message(
        f"Validation | Loss: {val_losses['total']:.4f} | "
        f"Det: {val_losses['detector']:.4f} "
        f"(Score: {val_losses['detector_score']:.4f}, Orient: {val_losses['detector_orient']:.4f}) | "
        f"Desc: {val_losses['descriptor']:.4f}",
        log_file,
    )

    return val_losses


def main():
    parser = argparse.ArgumentParser(description="Train ViT feature extractor")

    # Data arguments
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to HPatches dataset root",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[1200, 1600],
        help="Target image size (H W)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Loss weights
    parser.add_argument(
        "--lambda-det",
        type=float,
        default=1.0,
        help="Detector loss weight (includes orientation)",
    )
    parser.add_argument(
        "--lambda-desc", type=float, default=1.0, help="Descriptor loss weight"
    )
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument(
        "--alpha-orient",
        type=float,
        default=0.5,
        help="Orientation loss weight within detector loss",
    )

    # Dataset augmentation arguments
    parser.add_argument(
        "--pair-mode",
        type=str,
        default="reference_only",
        choices=["reference_only", "consecutive", "all_pairs"],
        help="Image pairing strategy",
    )
    parser.add_argument(
        "--use-synthetic-aug",
        action="store_true",
        default=False,
        help="Enable synthetic homography augmentation",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.5,
        help="Ratio of synthetic samples (0.0-1.0)",
    )
    parser.add_argument(
        "--synthetic-rotation-range",
        type=float,
        default=30.0,
        help="Maximum rotation in degrees for synthetic augmentation",
    )
    parser.add_argument(
        "--synthetic-scale-range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        help="Scale range (min max) for synthetic augmentation",
    )
    parser.add_argument(
        "--synthetic-perspective-range",
        type=float,
        default=0.0002,
        help="Perspective distortion range for synthetic augmentation",
    )
    parser.add_argument(
        "--synthetic-translation-range",
        type=float,
        nargs=2,
        default=[0.1, 0.1],
        help="Translation range (tx_ratio ty_ratio) for synthetic augmentation",
    )

    # Sampler arguments
    parser.add_argument(
        "--top-k", type=int, default=512, help="Number of invariant points"
    )
    parser.add_argument(
        "--negative-radius", type=int, default=16, help="Negative sampling radius"
    )
    parser.add_argument(
        "--num-negatives", type=int, default=10, help="Number of in-image negatives"
    )
    parser.add_argument(
        "--num-hard-negatives", type=int, default=5, help="Number of hard negatives"
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="top_k",
        choices=["top_k", "threshold", "hybrid"],
        help="Invariant point selection strategy",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for threshold-based selection",
    )
    parser.add_argument(
        "--min-invariant-points",
        type=int,
        default=100,
        help="Minimum number of invariant points (for threshold mode)",
    )
    parser.add_argument(
        "--max-invariant-points",
        type=int,
        default=2048,
        help="Maximum number of invariant points (for threshold mode)",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        help="DINOv2 backbone model",
    )
    parser.add_argument(
        "--descriptor-dim", type=int, default=128, help="Descriptor dimension"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--save-interval", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Logging arguments
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log every N batches"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), help="Log directory"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if None)",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=5,
        help="Generate loss plots every N epochs (0 to disable, default: 5)",
    )

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Performance optimization arguments
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision (AMP) for faster training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile() for model optimization (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        default=False,
        help="Enable cuDNN benchmark mode for faster training",
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Enable cuDNN benchmark mode if requested
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled")

    # Setup experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"vit_features_{timestamp}"

    # Setup logging
    log_file = setup_logging(args.log_dir, args.experiment_name)
    log_message(f"Starting experiment: {args.experiment_name}", log_file)
    log_message(f"Arguments: {args}", log_file)
    log_message(f"AMP enabled: {args.use_amp}", log_file)
    log_message(f"torch.compile enabled: {args.compile}", log_file)
    log_message(f"cuDNN benchmark enabled: {args.cudnn_benchmark}", log_file)

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Import modules after path setup
    from vit_colmap.model import ViTFeatureModel
    from vit_colmap.dataloader import (
        HPatchesDataset,
        TrainingSampler,
        TrainingBatchProcessor,
        collate_fn,
    )
    from vit_colmap.dataloader.synthetic_homography import SyntheticHomographyConfig
    from vit_colmap.losses import TotalLoss
    from vit_colmap.utils.plot_training import TrainingLossPlotter

    # Initialize model
    log_message("Initializing model...", log_file)
    model = ViTFeatureModel(
        backbone_name=args.backbone,
        descriptor_dim=args.descriptor_dim,
        freeze_backbone=True,
    )
    model.to(device)

    param_counts = model.count_parameters()
    log_message(f"Model parameters: {param_counts}", log_file)

    # Apply torch.compile if requested (PyTorch 2.0+)
    if args.compile:
        log_message("Applying torch.compile() optimization...", log_file)
        try:
            model = torch.compile(model)
            log_message("torch.compile() applied successfully", log_file)
        except Exception as e:
            log_message(
                f"torch.compile() failed: {e}. Continuing without it.", log_file
            )

    # Initialize gradient scaler for AMP
    scaler = (
        GradScaler(device=device.type)
        if args.use_amp and device.type == "cuda"
        else None
    )
    if scaler is not None:
        log_message("GradScaler initialized for AMP training", log_file)

    # Initialize dataset
    log_message(f"Loading HPatches dataset from {args.data_root}...", log_file)

    # Create synthetic augmentation config
    synthetic_config = None
    if args.use_synthetic_aug:
        synthetic_config = SyntheticHomographyConfig(
            rotation_range=args.synthetic_rotation_range,
            scale_range=tuple(args.synthetic_scale_range),
            perspective_range=args.synthetic_perspective_range,
            translation_range=tuple(args.synthetic_translation_range),
        )
        log_message(
            f"Synthetic augmentation enabled: {synthetic_config.to_dict()}", log_file
        )

    full_dataset = HPatchesDataset(
        root_dir=args.data_root,
        split="all",
        target_size=tuple(args.target_size),
        patch_size=model.patch_size,
        pair_mode=args.pair_mode,
        use_synthetic_aug=args.use_synthetic_aug,
        synthetic_ratio=args.synthetic_ratio,
        synthetic_config=synthetic_config,
    )

    # Split into train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    log_message(f"Train samples: {n_train}, Val samples: {n_val}", log_file)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Initialize sampler and batch processor
    sampler = TrainingSampler(
        top_k_invariant=args.top_k,
        negative_radius=args.negative_radius,
        num_in_image_negatives=args.num_negatives,
        num_hard_negatives=args.num_hard_negatives,
        patch_size=model.patch_size,
        selection_mode=args.selection_mode,
        similarity_threshold=args.similarity_threshold,
        min_invariant_points=args.min_invariant_points,
        max_invariant_points=args.max_invariant_points,
    )

    log_message("TrainingSampler configuration:", log_file)
    log_message(f"  Selection mode: {args.selection_mode}", log_file)
    if args.selection_mode in ["threshold", "hybrid"]:
        log_message(f"  Similarity threshold: {args.similarity_threshold}", log_file)
        log_message(f"  Min points: {args.min_invariant_points}", log_file)
        log_message(f"  Max points: {args.max_invariant_points}", log_file)
    if args.selection_mode in ["top_k", "hybrid"]:
        log_message(f"  Top K: {args.top_k}", log_file)

    processor = TrainingBatchProcessor(model, sampler)

    # Initialize loss function
    loss_fn = TotalLoss(
        lambda_det=args.lambda_det,
        lambda_desc=args.lambda_desc,
        margin=args.margin,
        alpha_orient=args.alpha_orient,
    )

    # Initialize optimizer
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Initialize loss plotter
    plotter = None
    if args.plot_interval > 0:
        plotter = TrainingLossPlotter(log_file=log_file)
        log_message(
            f"Loss plotting enabled (interval: {args.plot_interval} epochs)", log_file
        )

    # Resume from checkpoint if specified
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        log_message(f"Resuming from checkpoint: {args.resume}", log_file)
        start_epoch, global_step, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

        # CRITICAL FIX: Ensure scheduler's last_epoch matches the resumed epoch
        # This prevents the cosine curve from restarting and causing LR oscillation
        if scheduler.last_epoch != start_epoch:
            log_message(
                f"⚠️  Scheduler last_epoch ({scheduler.last_epoch}) != checkpoint epoch ({start_epoch})",
                log_file,
            )
            log_message(
                f"   Manually setting scheduler.last_epoch = {start_epoch}", log_file
            )
            scheduler.last_epoch = start_epoch

        start_epoch += 1
        log_message(f"Resumed from epoch {start_epoch - 1}", log_file)
        log_message(f"Scheduler last_epoch: {scheduler.last_epoch}", log_file)
        log_message(f"Current LR: {scheduler.get_last_lr()[0]:.6f}", log_file)

        # Run scheduler diagnostics to verify state
        diagnose_scheduler(scheduler, start_epoch - 1, log_file)

    # Training loop
    log_message("Starting training...", log_file)

    for epoch in range(start_epoch, args.epochs + 1):
        log_message(f"\n{'='*60}", log_file)
        log_message(f"Epoch {epoch}/{args.epochs}", log_file)
        log_message(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}", log_file)
        log_message(f"Scheduler last_epoch: {scheduler.last_epoch}", log_file)
        log_message(f"{'='*60}", log_file)

        # Train
        train_losses = train_one_epoch(
            model,
            train_loader,
            processor,
            loss_fn,
            optimizer,
            scheduler,
            device,
            epoch,
            args.log_interval,
            log_file,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        # Validate
        val_losses = validate(
            model,
            val_loader,
            processor,
            loss_fn,
            device,
            log_file,
            use_amp=args.use_amp,
        )

        # Update scheduler
        old_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        new_lr = scheduler.get_last_lr()[0]

        # Detect abnormal LR changes (should only decrease or stay similar)
        lr_ratio = new_lr / old_lr if old_lr > 0 else 1.0
        if lr_ratio > 1.5:  # LR increased by more than 50%
            log_message(
                f"⚠️  WARNING: Learning rate jumped from {old_lr:.6f} to {new_lr:.6f} (ratio: {lr_ratio:.2f})",
                log_file,
            )
            log_message("   This suggests a scheduler bug!", log_file)

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = args.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                train_losses,
                checkpoint_path,
            )
            log_message(f"Saved checkpoint: {checkpoint_path}", log_file)

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_path = args.checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, val_losses, best_path
            )
            log_message(
                f"New best model saved (val_loss: {best_val_loss:.4f})", log_file
            )

        # Save latest checkpoint
        latest_path = args.checkpoint_dir / "latest.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, train_losses, latest_path
        )

        # Generate loss plots at specified intervals
        if plotter is not None and epoch % args.plot_interval == 0:
            log_message("Generating loss plots...", log_file)
            try:
                plotter.clear_cache()  # Force reload to include latest log data
                plot_dir = args.log_dir / "plots"
                plotter.plot_all(output_dir=plot_dir, show=False)
                log_message(f"Loss plots saved to {plot_dir}", log_file)
            except Exception as e:
                log_message(f"Warning: Failed to generate plots: {e}", log_file)

    log_message("\nTraining complete!", log_file)
    log_message(f"Best validation loss: {best_val_loss:.4f}", log_file)
    log_message(f"Checkpoints saved to: {args.checkpoint_dir}", log_file)

    # Generate final plots
    if plotter is not None:
        log_message("\nGenerating final loss plots...", log_file)
        try:
            plotter.clear_cache()
            plot_dir = args.log_dir / "plots"
            results = plotter.plot_all(output_dir=plot_dir, show=False)
            for plot_name, plot_path in results.items():
                if plot_path:
                    log_message(f"  - {plot_name}: {plot_path}", log_file)
        except Exception as e:
            log_message(f"Warning: Failed to generate final plots: {e}", log_file)


if __name__ == "__main__":
    main()
