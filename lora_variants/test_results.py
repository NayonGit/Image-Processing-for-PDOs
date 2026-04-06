import torch
import argparse
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from lora_variants.data_rcnn import OrganoidDetectionDataset
from lora_variants.models import OrganoidDetectionModule

def evaluate_on_test_set(model_module, dataset_test, batch_size=4, device="cuda"):
    """
    Computes mAP metrics on a specific test dataset.
    """
    model_module.to(device)
    model_module.eval()
    
    # Initialize the metric (COCO standards: xyxy format)
    test_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
    
    # Setup DataLoader (small batch size recommended for VRAM safety)
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        collate_fn=lambda x: tuple(zip(*x)), # Required for Faster R-CNN
        num_workers=4,
        pin_memory=(device == "cuda")
    )
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Evaluation Progress"):
            # Move data to target device (GPU or CPU)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Model Inference
            outputs = model_module(images)
            
            # Move results back to CPU for metric calculation (prevents GPU memory clutter)
            outputs = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]
            targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
            
            test_metric.update(outputs, targets)
            
    # Final computation
    results = test_metric.compute()
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate an Organoid Detection model on a test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt lightning checkpoint")
    parser.add_argument("--dataset", type=str, default="tellu", choices=["tellu", "orgaquant", "multiorg"], help="Test dataset name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    
    args = parser.parse_args()

    print(f"\n--- Initializing Evaluation ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset   : {args.dataset}")
    print(f"Device    : {args.device}")

    try:
        dataset_test = OrganoidDetectionDataset(dataset_name=args.dataset, split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # The module automatically reconstructs the PEFT architecture from the checkpoint
    try:
        model_module = OrganoidDetectionModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    test_results = evaluate_on_test_set(model_module, dataset_test, batch_size=args.batch_size, device=args.device)

    print("\n" + "="*45)
    print(f" TEST RESULTS SUMMARY: {args.dataset.upper()}")
    print("="*45)
    print(f" mAP @ 0.50 (IoU)       : {test_results['map_50']:.4f}")
    print(f" mAP (0.50:0.95)        : {test_results['map']:.4f}")
    print(f" mAP @ 0.75 (Strict)    : {test_results['map_75']:.4f}")
    print(f" Recall (AR @ 100)      : {test_results['mar_100']:.4f}")
    print(f" Average Precision Small: {test_results['map_small']:.4f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()