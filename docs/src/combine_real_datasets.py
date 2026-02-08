"""
Combine real images from both WM811K datasets.
Use standard WM811K classes since short_bridge and open_circuit 
are not standard semiconductor defect types.
"""

import shutil
from pathlib import Path
from collections import Counter

# Sources
WM811K_DIR = Path("data/real_images/WM811k_Dataset")  # First dataset 
MIXED_DIR = Path("data/processed")  # Already processed mixed-type

# Destination (we'll update in place)
OUTPUT_DIR = Path("data/processed")

# Standard WM811K-based classes (6 classes instead of 8)
# The hackathon asks for 8, but WM811K only has these standard types
FINAL_CLASSES = ['clean', 'scratch', 'center_defect', 'edge_defect', 
                 'random_defect', 'other']

# WM811K source mapping
WM811K_MAP = {
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


def main():
    print("=" * 60)
    print("Combining Real Wafer Image Datasets")
    print("=" * 60)
    
    # Count existing images from mixed-type processing
    counts = Counter()
    for cls in FINAL_CLASSES:
        cls_dir = OUTPUT_DIR / cls
        if cls_dir.exists():
            counts[cls] = len(list(cls_dir.glob("*.png")))
    
    print("\nExisting images (from Mixed-type):")
    for cls in FINAL_CLASSES:
        print(f"  {cls}: {counts[cls]}")
    
    # Add clean images from WM811K
    print("\nAdding 'clean' (none) images from WM811K...")
    none_dir = WM811K_DIR / "none"
    if none_dir.exists():
        clean_dir = OUTPUT_DIR / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        for img_file in none_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                new_name = f"clean_{counts['clean']:04d}{img_file.suffix}"
                shutil.copy2(img_file, clean_dir / new_name)
                counts['clean'] += 1
                if counts['clean'] >= 150:
                    break
    
    print("\n" + "=" * 60)
    print("Final class distribution:")
    total = 0
    for cls in FINAL_CLASSES:
        print(f"  {cls}: {counts[cls]}")
        total += counts[cls]
    
    print(f"\nTotal: {total} real images")
    print(f"\nClasses: {len(FINAL_CLASSES)}")
    
    # Remove empty directories
    for cls in ['short_bridge', 'open_circuit']:
        cls_dir = OUTPUT_DIR / cls
        if cls_dir.exists() and not list(cls_dir.glob("*")):
            shutil.rmtree(cls_dir)
            print(f"Removed empty directory: {cls}")


if __name__ == "__main__":
    main()
