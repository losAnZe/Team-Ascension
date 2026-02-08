"""
Evaluation script for wafer defect detection model.
Generates detailed metrics and confusion matrix.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_data_loaders, CLASSES, IDX_TO_CLASS
from model import create_model


def evaluate_model(model, loader, device):
    """Run evaluation and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_confusion_matrix(y_true, y_pred, class_names):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Print header
    print("\nConfusion Matrix:")
    print("-" * 80)
    
    # Abbreviated class names for display
    short_names = [name[:8] for name in class_names]
    
    # Header row
    header = "         " + "".join([f"{n:>9}" for n in short_names])
    print(header)
    print("-" * 80)
    
    # Data rows
    for i, row in enumerate(cm):
        row_str = f"{short_names[i]:<9}" + "".join([f"{val:>9}" for val in row])
        print(row_str)
    
    print("-" * 80)
    return cm


def evaluate(args):
    """Main evaluation function."""
    print("=" * 60)
    print("Wafer Defect Detection - Model Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(num_classes=len(CLASSES))
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc']*100:.1f}%")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=64
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Confusion matrix
    cm = print_confusion_matrix(y_true, y_pred, CLASSES)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed metrics per class
    class_metrics = {}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, cls in enumerate(CLASSES):
        class_metrics[cls] = {
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1_score': float(f[i]),
            'support': int(s[i])
        }
    
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_samples': len(y_true)
        },
        'per_class': class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': CLASSES
    }
    
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate wafer defect model')
    parser.add_argument('--checkpoint', type=str, 
                        default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for metrics')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
