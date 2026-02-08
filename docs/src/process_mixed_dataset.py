"""
Process Mixed-type Wafer Defect dataset into our format.
This dataset has 38,015 real wafer images with 8 classes!
"""

import numpy as np
from PIL import Image
from pathlib import Path

# Paths
NPZ_FILE = Path("data/mixed_type/Wafer_Map_Datasets.npz")
OUTPUT_DIR = Path("data/processed")

# Classes in the mixed-type dataset (from documentation)
# 0: Center, 1: Donut, 2: Edge-Loc, 3: Edge-Ring, 4: Loc, 5: Near-full, 6: Scratch, 7: Random  
# We need to map these to our classes
CLASSES_SRC = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Scratch', 'Random']

CLASSES_DST = ['clean', 'scratch', 'center_defect', 'edge_defect', 
               'random_defect', 'short_bridge', 'open_circuit', 'other']

# Mapping source classes to destination
CLASS_MAP = {
    0: 'center_defect',   # Center
    1: 'other',           # Donut 
    2: 'edge_defect',     # Edge-Loc
    3: 'edge_defect',     # Edge-Ring
    4: 'random_defect',   # Loc
    5: 'other',           # Near-full
    6: 'scratch',         # Scratch
    7: 'random_defect',   # Random
}

# Target images per class
TARGET_PER_CLASS = 150


def main():
    print("=" * 60)
    print("Processing Mixed-type Wafer Defect Dataset")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {NPZ_FILE}...")
    data = np.load(NPZ_FILE, allow_pickle=True)
    images = data['arr_0']  # Shape: (38015, 52, 52)
    labels = data['arr_1']  # Shape: (38015, 8) - one-hot encoded
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    
    # Convert one-hot to class indices
    label_indices = np.argmax(labels, axis=1)
    
    # Create output directories
    for cls in CLASSES_DST:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {cls: 0 for cls in CLASSES_DST}
    
    # Process images
    print("\nProcessing images...")
    for i, (img_array, label_idx) in enumerate(zip(images, label_indices)):
        dst_class = CLASS_MAP[label_idx]
        
        # Skip if we have enough of this class
        if counts[dst_class] >= TARGET_PER_CLASS:
            continue
        
        # Convert to image
        # Normalize to 0-255 range
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(img_normalized, mode='L')
        
        # Resize to 64x64 for consistency
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Save
        filename = f"{dst_class}_{counts[dst_class]:04d}.png"
        img.save(OUTPUT_DIR / dst_class / filename)
        counts[dst_class] += 1
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1} images: {dict(counts)}")
    
    print("\n" + "=" * 60)
    print("Final class distribution:")
    total = 0
    for cls in CLASSES_DST:
        print(f"  {cls}: {counts[cls]}")
        total += counts[cls]
    
    print(f"\nTotal: {total} images")
    
    # Check for missing classes
    missing = [cls for cls in CLASSES_DST if counts[cls] == 0]
    if missing:
        print(f"\nNote: Missing classes: {missing}")
        print("The Mixed-type dataset doesn't have 'clean', 'short_bridge', or 'open_circuit'")
        print("These are not standard WM811K defect types.")


if __name__ == "__main__":
    main()
