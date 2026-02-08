"""
Reorganize to 8 classes using standard WM811K class names.
Instead of merging edge types and random types, keep them separate.

WM811K Standard Classes:
1. Center
2. Donut  
3. Edge-Loc
4. Edge-Ring
5. Loc (Local)
6. Near-full
7. None (Clean)
8. Scratch
9. Random

We'll use 8 of these for our model.
"""

import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

# Sources
WM811K_DIR = Path("data/real_images/WM811k_Dataset")
MIXED_NPZ = Path("data/mixed_type/Wafer_Map_Datasets.npz")

# Output
OUTPUT_DIR = Path("data/processed")

# Target classes - standard WM811K names
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

TARGET_PER_CLASS = 100


def main():
    print("=" * 60)
    print("Creating 8-Class Real Dataset")
    print("=" * 60)
    
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {cls: 0 for cls in CLASSES}
    
    # 1. Process Mixed-type NPZ (has labeled images)
    print("\n1. Processing Mixed-type dataset...")
    data = np.load(MIXED_NPZ, allow_pickle=True)
    images = data['arr_0']  # (38015, 52, 52)
    labels = data['arr_1']  # (38015, 8) one-hot
    label_indices = np.argmax(labels, axis=1)
    
    # Mixed-type classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random
    mixed_map = {
        0: 'center',
        1: 'donut',
        2: 'edge_loc',
        3: 'edge_ring',
        4: 'loc',
        5: 'donut',     # Near-full -> donut (similar pattern)
        6: 'scratch',
        7: 'random'
    }
    
    for i, (img_array, label_idx) in enumerate(zip(images, label_indices)):
        dst_class = mixed_map[label_idx]
        
        if counts[dst_class] >= TARGET_PER_CLASS:
            continue
        
        # Convert to image
        img_norm = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(img_norm, mode='L')
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        filename = f"{dst_class}_{counts[dst_class]:04d}.png"
        img.save(OUTPUT_DIR / dst_class / filename)
        counts[dst_class] += 1
    
    print(f"  After Mixed-type: {dict(counts)}")
    
    # 2. Add 'none' (clean) from WM811K images
    print("\n2. Adding 'none' (clean) from WM811K...")
    none_dir = WM811K_DIR / "none"
    if none_dir.exists():
        for img_file in sorted(none_dir.glob("*"))[:TARGET_PER_CLASS]:
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Load and resize
                img = Image.open(img_file).convert('L')
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                filename = f"none_{counts['none']:04d}.png"
                img.save(OUTPUT_DIR / 'none' / filename)
                counts['none'] += 1
    
    print(f"  Final: {dict(counts)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Final Dataset:")
    print("=" * 60)
    total = 0
    for cls in CLASSES:
        print(f"  {cls}: {counts[cls]}")
        total += counts[cls]
    print(f"\nTotal: {total} real images")
    print(f"Classes: {len(CLASSES)}")


if __name__ == "__main__":
    main()
