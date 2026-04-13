import argparse
import os
import lightning as L
import torch
from CenterNet.train_centernet import run_experiment, run_test


def get_shared_parser():
    """Shared arguments for both training and testing."""
    parser = argparse.ArgumentParser(add_help=False)
    
    # Data
    parser.add_argument("--dataset", type=str, default="tellu",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Nom du dataset dans DATASET_INFO")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size")
    parser.add_argument("--downsample", type = float, default = 3.5, help="Choose downsampling side")
    
    # Model & PEFT
    parser.add_argument("--name", type=str, default="dinov2", choices=["dinov2", "dinov3"], help="Choose a model name")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base","large","giant"], help="Choose a model size")
    parser.add_argument("--method", type=str, default="lora", 
                        choices=["frozen", "lora", "dora"],
                        help="Fine-tuning method: 'frozen' for frozen backbone, 'lora' for LoRA, 'dora' for DoRA")
    parser.add_argument("--rank", type=int, default=8, help="Rank LoRA/DoRA")
    
    # Output configuration
    parser.add_argument("--output_root", type=str, default="centernet_models",
                        help="Root directory for saving models and logs")
    
    return parser

def main():
    print(f"🚀 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    root_parser = argparse.ArgumentParser(description="CLI for CenterNet + DINOv2/v3 experiments")
    subparsers = root_parser.add_subparsers(dest="command", help="Command to execute")

    train_parser = subparsers.add_parser("train", parents=[get_shared_parser()], help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the detection head")
    train_parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone (LoRA)")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Path to a .ckpt file to resume from")

    test_parser = subparsers.add_parser("test", parents=[get_shared_parser()], help="Test a trained model")
    test_parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the best checkpoint .ckpt")

    args = root_parser.parse_args()

    if args.command is None:
        root_parser.print_help()
        return

    exp_name = f"{args.method}_r{args.rank}" if args.method != "frozen" else "frozen_backbone"
    
    if args.command == "train":
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {exp_name.upper()} on {args.dataset.upper()}")
        print(f"Resume from: {args.resume_from if args.resume_from else 'None (Fresh start)'}")        
        print(f"{'='*60}\n")

        trainer = run_experiment(
            name = args.name,
            model_size = args.model_size,
            method=args.method,
            dataset_name=args.dataset,
            r=args.rank,
            lr = args.lr,
            lr_backbone = args.lr_backbone,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patch_size = args.patch_size,
            downsample= args.downsample,
            resume_from_checkpoint=args.resume_from,
        )

        print(f"\n[Done] Experiment {exp_name} finished.")
    
    elif args.command == "test":
        print(f"\n{'='*40}\n🎯 TEST (Final Evaluation)\n{'='*40}")
        metrics = run_test(
            ckpt_path=args.ckpt_path,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            downsample=args.downsample,
        )
        print("\nTest Results :")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()