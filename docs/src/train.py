"""
Training script for wafer defect detection model.
Optimized for CPU training with early stopping.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_data_loaders, CLASSES
from model import create_model, count_parameters


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_class_weights(train_loader, num_classes, device):
    """Calculate class weights for imbalanced data."""
    class_counts = torch.zeros(num_classes)
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    
    return weights.to(device)


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Wafer Defect Detection - Model Training")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Model
    print("\nCreating model...")
    model = create_model(num_classes=len(CLASSES))
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss with class weights
    class_weights = get_class_weights(train_loader, len(CLASSES), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'checkpoints' / 'best_model.pth')
            print(f"  [*] New best model saved! (Val Acc: {val_acc*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping triggered after {args.patience} epochs without improvement")
                break
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, output_dir / 'checkpoints' / 'latest_model.pth')
    
    training_time = time.time() - start_time
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}% (epoch {best_epoch})")
    print(f"\nModel saved to: {output_dir / 'checkpoints' / 'best_model.pth'}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'checkpoints' / 'best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    
    # Save final results
    results = {
        'model': 'WaferDefectCNN',
        'num_params': num_params,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time_minutes': training_time / 60,
        'device': str(device),
        'epochs_trained': epoch,
        'classes': CLASSES,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'training_results.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train wafer defect detection model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Input image size')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
