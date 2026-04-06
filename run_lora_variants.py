import argparse
import os
import lightning as L
import torch

from lora_variants.train import run_experiment

torch.set_float32_matmul_precision('high')
def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN with PEFT variants on Organoids")
    
    parser.add_argument("--method", type=str, default="lora", 
                        choices=["full", "lora", "dora"],
                        help="The fine-tuning method to use")
    parser.add_argument("--dataset", type=str, default="tellu",
                        choices=["tellu", "orgaquant", "multiorg"],
                        help="Name of the dataset in DATASET_INFO")
    parser.add_argument("--run-num", type=int, default=0, help="Optional run number for logging")
    
    # Hyperparameters
    parser.add_argument("--rank", type=int, default=16, help="Rank for LoRA/DoRA/AdaLoRA")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Output configuration
    parser.add_argument("--output_root", type=str, default="rcnn_models",
                        help="Root directory to save models and logs")

    args = parser.parse_args()

    # Example : rcnn_models/tellu/dora_r16_0/
    exp_name = f"{args.method}_r{args.rank}_{args.run_num}" if args.method != "full" else f"full_finetuning_{args.run_num}"
    output_dir = os.path.join(args.output_root, args.dataset, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {exp_name.upper()} on {args.dataset.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    trainer = run_experiment(
        method=args.method,
        dataset_name=args.dataset,
        r=args.rank,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        output_dir=output_dir,
    )

    print(f"\n[Done] Experiment {exp_name} finished.")
    print(f"Checkpoints and logs are in: {output_dir}")

if __name__ == "__main__":
    main()