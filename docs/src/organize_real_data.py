"""
Organize real WM811K images into our 8-class format.
"""

import shutil
from pathlib import Path

# Source and destination
SRC_DIR = Path("data/real_images/WM811k_Dataset")
DST_DIR = Path("data/processed")

# Class mapping from WM811K to our classes
CLASS_MAP = {
    'none': 'clean',
    'Center': 'center_defect',
    'Edge Local': 'edge_defect',
    'Edge Ring': 'edge_defect',
    'Local': 'random_defect',
    'random': 'random_defect',
    'Scratch': 'scratch',
    'Donut': 'other',
    'near full': 'other',
}

# Our target classes
CLASSES = ['clean', 'scratch', 'center_defect', 'edge_defect', 
           'random_defect', 'short_bridge', 'open_circuit', 'other']


def main():
    print("=" * 60)
    print("Organizing Real WM811K Images")
    print("=" * 60)
    
    # Clean existing processed data
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    
    # Create output directories
    for cls in CLASSES:
        (DST_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {cls: 0 for cls in CLASSES}
    
    # Copy and rename images
    for src_class, dst_class in CLASS_MAP.items():
        src_path = SRC_DIR / src_class
        if not src_path.exists():
            print(f"Warning: {src_path} does not exist, skipping")
            continue
        
        for img_file in src_path.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # New filename
                new_name = f"{dst_class}_{counts[dst_class]:04d}{img_file.suffix}"
                dst_path = DST_DIR / dst_class / new_name
                
                # Copy file
                shutil.copy2(img_file, dst_path)
                counts[dst_class] += 1
    
    print("\nImages copied:")
    total = 0
    for cls in CLASSES:
        print(f"  {cls}: {counts[cls]}")
        total += counts[cls]
    
    print(f"\nTotal: {total} images")
    print(f"Saved to: {DST_DIR}")
    
    # Note about missing classes
    missing = [cls for cls in CLASSES if counts[cls] == 0]
    if missing:
        print(f"\nNote: Classes with no images: {missing}")
        print("These will need to be supplemented (e.g., with synthetic data)")


if __name__ == "__main__":
    main()
