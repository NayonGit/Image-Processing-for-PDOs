#!/usr/bin/env python3

import argparse
import os
import torch
import lightning as L

from FasterRCNN.train_rcnn import run_train, run_test

"""
This file contains the main CLI for running training and testing for the Faster R-CNN + various backbones project.
It defines a main function that parses command-line arguments for both training and testing, 
and then calls the appropriate functions from train_rcnn.py to execute the desired action.
"""

# Optimization TensorCore
torch.set_float32_matmul_precision('high')

def get_train_parser():
    """
    Parser for the training command.
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    # Backbone Configuration
    parser.add_argument("--backbone", type=str, default="rcnn", 
                        choices=["rcnn", "dinov2", "swin", "convnextv2"],
                        help="Backbone architecture to use")
    
    #  PEFT Configuration
    parser.add_argument("--method", type=str, default="lora", 
                        choices=["full", "lora", "dora"],
                        help="Fine-tuning method")
    parser.add_argument("--rank", type=int, default=8, 
                        help="Rank for LoRA/DoRA (r)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="tellu",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Dataset to use for training")
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Logging and Resuming
    parser.add_argument("--run-num", type=int, default=0, help="Number of the experiment")
    parser.add_argument("--output-root", type=str, default="rcnn_models",
                        help="Root directory for saving models and logs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint (.ckpt) to resume from")
    
    return parser

def get_test_parser():
    """Parser for the testing command."""
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to the .ckpt weight file")
    parser.add_argument("--dataset", type=str, default="tellu",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Dataset to use for testing")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-root", type=str, default="test_results")
    
    parser.add_argument("--backbone", type=str, default="rcnn")
    parser.add_argument("--method", type=str, default="lora")
    parser.add_argument("--rank", type=int, default=8)

    return parser

def main():
    # Setup Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    root_parser = argparse.ArgumentParser(
        description="Training and testing script for Faster R-CNN variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = root_parser.add_subparsers(dest="command", help="Command to execute")

    # Subcommand Train:
    train_parser = subparsers.add_parser("train", help="Train a model", parents=[get_train_parser()])
    
    # Subcommand Test:
    test_parser = subparsers.add_parser("test", help="Test a model", parents=[get_test_parser()])

    args = root_parser.parse_args()

    if args.command is None:
        root_parser.print_help()
        return


    if args.command == "train":    
        exp_name = f"{args.method}_r{args.rank}_{args.run_num}" if args.method != "full" else f"full_finetuning_{args.run_num}"
        output_dir = os.path.join(args.output_root, args.dataset, exp_name)
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print(f"🚀 TRAINING: {args.backbone.upper()} | {args.method.upper()} | r={args.rank}")
        print(f"📍 Output: {output_dir}")
        print("=" * 80)
        
        run_train(
            backbone=args.backbone,
            method=args.method,
            dataset_name=args.dataset,
            r=args.rank,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            output_dir=output_dir,
            resume_from=args.resume_from
        )

    elif args.command == "test":
        ckpt_name = os.path.basename(args.ckpt_path).replace('.ckpt', '')
        output_dir = os.path.join(args.output_root, args.dataset, ckpt_name)
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print(f"🧪 TESTING: {args.ckpt_path}")
        print(f"📊 Dataset: {args.dataset}")
        print(f"📍 Results: {output_dir}")
        print("=" * 80)

        run_test(
            ckpt_path=args.ckpt_path,
            backbone=args.backbone,
            method=args.method,
            dataset_name=args.dataset,
            r=args.rank,
            batch_size=args.batch_size,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()