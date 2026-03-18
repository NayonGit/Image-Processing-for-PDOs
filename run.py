#!/usr/bin/env python3
"""
Script CLI for training and testing object detection models (frozen backbone).

Usage:
    # Train a model
    python run.py train --dataset-name multiorg --batch-size 8 --max-epochs 100

    # Resume training from a checkpoint
    python run.py train --dataset-name multiorg --resume-from ./checkpoints/last.ckpt

    # Test a model
    python run.py test --ckpt-path ./checkpoints/best.ckpt --dataset-name multiorg
"""

import argparse
import os
import torch

from detection.train import train as run_train
from detection.train import test as run_test


def get_train_parser():
    """Parser for training command."""
    parser = argparse.ArgumentParser(description="Train a detection model with frozen backbone")
    
    # Data
    parser.add_argument("--dataset-name", type=str, default="multiorg",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Name of the dataset to train on")
    parser.add_argument("--h5-path", type=str, default=None,
                        help="Path to H5 dataset file (uses default if None)")
    parser.add_argument("--train-val-split", type=float, default=0.85,
                        help="Fraction of data to use for training")
    
    # Model
    parser.add_argument("--backbone-name", type=str, default="dinov2",
                        choices=["dinov2", "dinov3"],
                        help="Name of the backbone model")
    parser.add_argument("--backbone-size", type=str, default="base",
                        choices=["small", "base", "large", "giant"],
                        help="Size of the backbone model")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes (auto-inferred if None)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension of detection head")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of object queries in DETR")
    parser.add_argument("--num-decoder-heads", type=int, default=8,
                        help="Number of attention heads in decoder")
    parser.add_argument("--num-decoder-layers", type=int, default=6,
                        help="Number of transformer decoder layers")
    
    # Loss
    parser.add_argument("--cost-class", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--cost-bbox", type=float, default=5.0,
                        help="Bounding box L1 loss weight")
    parser.add_argument("--cost-giou", type=float, default=2.0,
                        help="GIoU loss weight")
    parser.add_argument("--eos-coef", type=float, default=0.1,
                        help="End-of-sequence coefficient")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    # Patching
    parser.add_argument("--use-patching", action="store_true", default=False,
                        help="Use patching for training")
    parser.add_argument("--num-patches", type=int, default=None,
                        help="Number of patches (auto-calculated if None)")
    parser.add_argument("--patch-size", type=int, default=224,
                        help="Patch size")
    parser.add_argument("--overlap-size", type=int, default=30,
                        help="Overlap size between patches")
    
    # Resume training
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    return parser


def get_test_parser():
    """Parser for testing command."""
    parser = argparse.ArgumentParser(description="Test a trained detection model")
    
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to checkpoint file")
    
    # Data
    parser.add_argument("--dataset-name", type=str, default="multiorg",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Name of the dataset to test on")
    parser.add_argument("--h5-path", type=str, default=None,
                        help="Path to H5 dataset file (uses default if None)")
    parser.add_argument("--train-val-split", type=float, default=0.85,
                        help="Fraction of data used for training (for consistency)")
    
    # Model
    parser.add_argument("--backbone-name", type=str, default="dinov2",
                        choices=["dinov2", "dinov3"],
                        help="Name of the backbone model")
    parser.add_argument("--backbone-size", type=str, default="base",
                        help="Size of the backbone model")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension of detection head")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of object queries in DETR")
    parser.add_argument("--num-decoder-heads", type=int, default=8,
                        help="Number of attention heads in decoder")
    parser.add_argument("--num-decoder-layers", type=int, default=6,
                        help="Number of transformer decoder layers")
    
    # Loss
    parser.add_argument("--cost-class", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--cost-bbox", type=float, default=5.0,
                        help="Bounding box L1 loss weight")
    parser.add_argument("--cost-giou", type=float, default=2.0,
                        help="GIoU loss weight")
    parser.add_argument("--eos-coef", type=float, default=0.1,
                        help="End-of-sequence coefficient")
    
    # Training (for model initialization)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save test results")
    
    # Patching
    parser.add_argument("--use-patching", action="store_true", default=False,
                        help="Use patching for testing")
    parser.add_argument("--num-patches", type=int, default=None,
                        help="Number of patches")
    parser.add_argument("--patch-size", type=int, default=224,
                        help="Patch size")
    parser.add_argument("--overlap-size", type=int, default=30,
                        help="Overlap size between patches")
    
    return parser


def main():
    """Main entry point."""
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Root parser
    root_parser = argparse.ArgumentParser(
        description="Training and testing script for object detection with frozen backbone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python run.py train --dataset-name multiorg --batch-size 8

  # Resume training from checkpoint
  python run.py train --dataset-name multiorg --resume-from ./checkpoints/last.ckpt

  # Test a model
  python run.py test --ckpt-path ./checkpoints/best.ckpt --dataset-name multiorg
        """
    )
    
    subparsers = root_parser.add_subparsers(dest="command", help="Command to run")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_args = get_train_parser()
    for action in train_args._actions:
        if action.dest != "help":
            train_parser._add_action(action)
    
    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test a model")
    test_args = get_test_parser()
    for action in test_args._actions:
        if action.dest != "help":
            test_parser._add_action(action)
    
    args = root_parser.parse_args()
    
    if args.command is None:
        root_parser.print_help()
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.command == "train":
        print("=" * 80)
        print("TRAINING MODEL (Frozen Backbone)")
        print("=" * 80)
        print(f"Dataset: {args.dataset_name}")
        print(f"Backbone: {args.backbone_name} ({args.backbone_size})")
        print(f"Batch size: {args.batch_size}")
        print(f"Max epochs: {args.max_epochs}")
        print(f"Use patching: {args.use_patching}")
        if args.resume_from:
            print(f"Resuming from: {args.resume_from}")
        print("=" * 80)
        print()
        
        metrics = run_train(
            dataset_name=args.dataset_name,
            h5_path=args.h5_path,
            train_val_split=args.train_val_split,
            backbone_name=args.backbone_name,
            backbone_size=args.backbone_size,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            num_decoder_heads=args.num_decoder_heads,
            num_decoder_layers=args.num_decoder_layers,
            cost_class=args.cost_class,
            cost_bbox=args.cost_bbox,
            cost_giou=args.cost_giou,
            eos_coef=args.eos_coef,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            output_dir=args.output_dir,
            use_patching=args.use_patching,
            num_patches=args.num_patches,
            patch_size=args.patch_size,
            overlap_size=args.overlap_size,
            resume_from_checkpoint=args.resume_from,
        )
        
        print()
        print("=" * 80)
        print("TRAINING FINISHED")
        print("=" * 80)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
    elif args.command == "test":
        print("=" * 80)
        print("TESTING MODEL (Frozen Backbone)")
        print("=" * 80)
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 80)
        print()
        
        metrics = run_test(
            ckpt_path=args.ckpt_path,
            dataset_name=args.dataset_name,
            h5_path=args.h5_path,
            train_val_split=args.train_val_split,
            backbone_name=args.backbone_name,
            backbone_size=args.backbone_size,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            num_queries=args.num_queries,
            num_decoder_heads=args.num_decoder_heads,
            num_decoder_layers=args.num_decoder_layers,
            cost_class=args.cost_class,
            cost_bbox=args.cost_bbox,
            cost_giou=args.cost_giou,
            eos_coef=args.eos_coef,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            output_dir=args.output_dir,
            use_patching=args.use_patching,
            num_patches=args.num_patches,
            patch_size=args.patch_size,
            overlap_size=args.overlap_size,
        )
        
        print()
        print("=" * 80)
        print("TESTING FINISHED")
        print("=" * 80)
        print("Metrics:")
        if isinstance(metrics, list):
            for i, metric_dict in enumerate(metrics):
                print(f"  Batch {i}:")
                for k, v in metric_dict.items():
                    print(f"    {k}: {v}")
        else:
            for k, v in metrics.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
