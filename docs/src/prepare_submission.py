"""
Reorganize dataset into Train/Validation/Test folder structure 
as required by hackathon submission guidelines.
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Paths
SRC_DIR = Path("data/processed")
DST_DIR = Path("data/submission")

# Classes
CLASSES = ['none', 'center', 'donut', 'edge_loc', 'edge_ring', 'loc', 'scratch', 'random']

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    print("=" * 60)
    print("Reorganizing Dataset for Submission")
    print("=" * 60)
    
    random.seed(42)
    
    # Clean destination
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    
    # Create folder structure
    for split in ['Train', 'Validation', 'Test']:
        for cls in CLASSES:
            (DST_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    total_counts = {'Train': 0, 'Validation': 0, 'Test': 0}
    
    # Process each class
    for cls in CLASSES:
        src_cls_dir = SRC_DIR / cls
        if not src_cls_dir.exists():
            print(f"Warning: {cls} not found")
            continue
        
        # Get all images
        images = list(src_cls_dir.glob("*.png"))
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(img, DST_DIR / 'Train' / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, DST_DIR / 'Validation' / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, DST_DIR / 'Test' / cls / img.name)
        
        total_counts['Train'] += len(train_imgs)
        total_counts['Validation'] += len(val_imgs)
        total_counts['Test'] += len(test_imgs)
        
        print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    for split, count in total_counts.items():
        print(f"  {split}: {count} images")
    print(f"  Total: {sum(total_counts.values())} images")
    print(f"\nSaved to: {DST_DIR}")


if __name__ == "__main__":
    main()
