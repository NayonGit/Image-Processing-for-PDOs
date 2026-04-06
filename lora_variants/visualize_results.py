
import argparse
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from lora_variants.data_rcnn import OrganoidDetectionDataset
from lora_variants.models import OrganoidDetectionModule



def visualize_and_save(model_module, dataset, output_path, device="cuda", threshold=0.5):

    """
    Selects a random image, runs inference, and saves the visualization.
    """

    model_module.to(device)
    model_module.eval()

    # Pick a random index and prepare data
    idx = random.randint(0, len(dataset) - 1)
    image, target = dataset[idx]

    # Add batch dimension [3, H, W] -> [1, 3, H, W]
    input_tensor = image.to(device).unsqueeze(0)

    # Model Inference
    with torch.no_grad():
        prediction = model_module(input_tensor)[0]

    # Processing for visualization
    # Convert image back to [H, W, C] and 0-1 range for plotting
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    h, w = image_np.shape[:2]

    gt_boxes = target["boxes"].cpu().numpy()
    if gt_boxes.size > 0 and gt_boxes.max() <= 1.01:
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h

    print(f"DEBUG: Shape GT boxes: {gt_boxes.shape}")
    if len(gt_boxes) > 0:
        print(f"DEBUG: First GT box values: {gt_boxes[0]}")   
    pred_boxes = prediction["boxes"].cpu().numpy()
    pred_scores = prediction["scores"].cpu().numpy()

    # Filter by confidence threshold
    keep = pred_scores > threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image_np)

    # Draw Ground Truth (GT) - Green
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='lime', facecolor='none', alpha=0.7)
        ax.add_patch(rect)

    # Draw Predictions - Red
    for box, score in zip(pred_boxes, pred_scores):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1]-5, f"{score:.2f}", color='white', weight='bold',
                fontsize=8, bbox=dict(facecolor='red', alpha=0.5, pad=0))

    plt.title(f"Model Inference | Dataset: {dataset.dataset_name} | Green: GT | Red: Preds")
    plt.axis('off')

    # Save the result
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions on a random image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--dataset", type=str, default="tellu", choices=["tellu", "orgaquant", "multiorg"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Where to save the PNG")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and model
    dataset = OrganoidDetectionDataset(dataset_name=args.dataset, split="test")
    model_module = OrganoidDetectionModule.load_from_checkpoint(args.checkpoint, map_location=args.device)

    # Filename based on checkpoint and dataset

    ckpt_name = os.path.basename(args.checkpoint).replace(".ckpt", "")
    output_filename = f"pred_{args.dataset}_{ckpt_name}.png"
    output_path = os.path.join(args.output_dir, output_filename)

    # Run visualization
    visualize_and_save(model_module, dataset, output_path, device=args.device, threshold=args.threshold)

if __name__ == "__main__":
    main()