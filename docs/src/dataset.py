"""
Dataset classes and data augmentation for wafer defect detection.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# Standard WM811K class names (8 classes)
CLASSES = [
    'none',        # Clean wafers
    'center',      # Center defect
    'donut',       # Donut pattern
    'edge_loc',    # Edge localized
    'edge_ring',   # Edge ring
    'loc',         # Local defect
    'scratch',     # Scratch
    'random'       # Random defects
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


class WaferDataset(Dataset):
    """PyTorch Dataset for wafer defect images."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of class indices
            transform: Optional torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load grayscale image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Ensure grayscale
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor and normalize
            img = T.ToTensor()(img)
        
        label = self.labels[idx]
        return img, label


def get_transforms(train=True, img_size=64):
    """Get data transforms for training/validation.
    
    Args:
        train: Whether to apply augmentation (True for training)
        img_size: Target image size
    
    Returns:
        torchvision.transforms.Compose
    """
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])


def load_dataset_paths(data_dir='data/processed'):
    """Load all image paths and labels from directory structure.
    
    Expected structure:
        data_dir/
            class_name/
                image1.png
                image2.png
                ...
    
    Returns:
        image_paths: List of Path objects
        labels: List of class indices
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        class_idx = CLASS_TO_IDX[class_name]
        
        for img_path in class_dir.glob('*.png'):
            image_paths.append(img_path)
            labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} images from {len(CLASSES)} classes")
    return image_paths, labels


def create_data_loaders(data_dir='data/processed', 
                         batch_size=32, 
                         img_size=64,
                         test_size=0.2,
                         val_size=0.1,
                         num_workers=0,
                         random_state=42):
    """Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to processed data
        batch_size: Batch size for training
        img_size: Image size
        test_size: Fraction for test set
        val_size: Fraction for validation (from train set)
        num_workers: DataLoader workers
        random_state: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load all data
    image_paths, labels = load_dataset_paths(data_dir)
    
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)  # Adjust for remaining data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    
    # Create datasets with transforms
    train_dataset = WaferDataset(train_paths, train_labels, 
                                  transform=get_transforms(train=True, img_size=img_size))
    val_dataset = WaferDataset(val_paths, val_labels,
                                transform=get_transforms(train=False, img_size=img_size))
    test_dataset = WaferDataset(test_paths, test_labels,
                                 transform=get_transforms(train=False, img_size=img_size))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
