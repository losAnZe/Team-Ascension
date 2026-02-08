"""
Synthetic Wafer Defect Dataset Generator
Generates grayscale wafer map images with realistic defect patterns.

This approach is used when the original LSWMD.pkl cannot be loaded
due to Python version incompatibility.

Defect patterns are based on real WM-811K failure types.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import random

# Configuration
OUTPUT_DIR = Path("data/processed")
IMG_SIZE = 64  # Wafer map resolution
WAFER_RADIUS = 28  # Radius of the circular wafer
IMAGES_PER_CLASS = 150  # Target images per class (150 * 8 = 1200 total)

# 8 defect classes as required by hackathon
CLASSES = [
    'clean',           # No defects
    'scratch',         # Linear scratch patterns
    'center_defect',   # Defects concentrated in center
    'edge_defect',     # Defects along edges
    'random_defect',   # Random scattered defects
    'short_bridge',    # Clusters indicating shorts/bridges
    'open_circuit',    # Line-like open circuit patterns  
    'other'            # Donut, near-full, mixed patterns
]


def create_wafer_mask(size=64, radius=28):
    """Create a circular wafer mask."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = ((x - center)**2 + (y - center)**2) <= radius**2
    return mask.astype(np.uint8)


def create_base_wafer(size=64, radius=28):
    """Create a base wafer with normal dies (value 128)."""
    wafer = np.zeros((size, size), dtype=np.uint8)
    mask = create_wafer_mask(size, radius)
    wafer[mask == 1] = 128  # Normal dies are gray
    return wafer, mask


def add_defects_clean(wafer, mask):
    """Clean wafer - no defects, just normal variation."""
    # Add slight random noise for realism
    noise = np.random.randint(-5, 6, wafer.shape).astype(np.int16)
    wafer = np.clip(wafer.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    wafer[mask == 0] = 0  # Keep background black
    return wafer


def add_defects_scratch(wafer, mask):
    """Add scratch pattern - linear defects."""
    # Random scratch angle and position
    angle = random.uniform(-np.pi/4, np.pi/4)
    center = wafer.shape[0] // 2
    
    for _ in range(random.randint(1, 3)):  # 1-3 scratches
        offset = random.randint(-15, 15)
        length = random.randint(20, 50)
        width = random.randint(1, 3)
        
        for i in range(-length//2, length//2):
            x = int(center + i * np.cos(angle) + offset)
            y = int(center + i * np.sin(angle))
            if 0 <= x < wafer.shape[0] and 0 <= y < wafer.shape[1]:
                for w in range(-width//2, width//2 + 1):
                    nx, ny = x + w, y + w
                    if 0 <= nx < wafer.shape[0] and 0 <= ny < wafer.shape[1]:
                        if mask[ny, nx]:
                            wafer[ny, nx] = 255  # Defect is white
    return wafer


def add_defects_center(wafer, mask):
    """Add center defect pattern - concentrated in middle."""
    center = wafer.shape[0] // 2
    defect_radius = random.randint(5, 12)
    intensity = random.uniform(0.3, 0.7)
    
    y, x = np.ogrid[:wafer.shape[0], :wafer.shape[1]]
    center_mask = ((x - center)**2 + (y - center)**2) <= defect_radius**2
    
    # Add defects in center region
    defect_mask = center_mask & (mask == 1) & (np.random.random(wafer.shape) < intensity)
    wafer[defect_mask] = 255
    
    return wafer


def add_defects_edge(wafer, mask):
    """Add edge defect pattern - defects along wafer edge."""
    center = wafer.shape[0] // 2
    inner_radius = WAFER_RADIUS - random.randint(3, 8)
    
    # Create ring at edge
    y, x = np.ogrid[:wafer.shape[0], :wafer.shape[1]]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Edge ring
    edge_mask = (dist >= inner_radius) & (dist <= WAFER_RADIUS) & (mask == 1)
    
    # Partial edge (arc)
    angle = np.arctan2(y - center, x - center)
    start_angle = random.uniform(0, np.pi)
    arc_length = random.uniform(np.pi/2, np.pi)
    angle_mask = (angle >= start_angle) & (angle <= start_angle + arc_length)
    
    defect_mask = edge_mask & angle_mask & (np.random.random(wafer.shape) < 0.6)
    wafer[defect_mask] = 255
    
    return wafer


def add_defects_random(wafer, mask):
    """Add random scattered defects."""
    defect_density = random.uniform(0.05, 0.15)
    defect_mask = (np.random.random(wafer.shape) < defect_density) & (mask == 1)
    wafer[defect_mask] = 255
    return wafer


def add_defects_short_bridge(wafer, mask):
    """Add cluster patterns representing shorts/bridges."""
    num_clusters = random.randint(2, 5)
    center = wafer.shape[0] // 2
    
    for _ in range(num_clusters):
        # Random cluster position
        cx = center + random.randint(-20, 20)
        cy = center + random.randint(-20, 20)
        cluster_size = random.randint(3, 8)
        
        # Draw cluster
        y, x = np.ogrid[:wafer.shape[0], :wafer.shape[1]]
        cluster_mask = ((x - cx)**2 + (y - cy)**2) <= cluster_size**2
        cluster_mask = cluster_mask & (mask == 1)
        wafer[cluster_mask] = 255
    
    return wafer


def add_defects_open_circuit(wafer, mask):
    """Add line patterns representing open circuits."""
    center = wafer.shape[0] // 2
    num_lines = random.randint(2, 4)
    
    for _ in range(num_lines):
        # Horizontal or vertical line segments
        if random.random() < 0.5:
            # Horizontal line
            y = center + random.randint(-20, 20)
            x_start = random.randint(5, 25)
            x_end = random.randint(40, 58)
            for x in range(x_start, x_end):
                if 0 <= x < wafer.shape[1] and 0 <= y < wafer.shape[0]:
                    if mask[y, x]:
                        wafer[y, x] = 255
                        if y + 1 < wafer.shape[0] and mask[y+1, x]:
                            wafer[y+1, x] = 255
        else:
            # Vertical line
            x = center + random.randint(-20, 20)
            y_start = random.randint(5, 25)
            y_end = random.randint(40, 58)
            for y in range(y_start, y_end):
                if 0 <= x < wafer.shape[1] and 0 <= y < wafer.shape[0]:
                    if mask[y, x]:
                        wafer[y, x] = 255
                        if x + 1 < wafer.shape[1] and mask[y, x+1]:
                            wafer[y, x+1] = 255
    
    return wafer


def add_defects_other(wafer, mask):
    """Add other patterns - donut, near-full, mixed."""
    pattern = random.choice(['donut', 'near_full', 'mixed'])
    center = wafer.shape[0] // 2
    
    if pattern == 'donut':
        # Ring pattern in middle
        inner_r = random.randint(5, 10)
        outer_r = random.randint(12, 18)
        y, x = np.ogrid[:wafer.shape[0], :wafer.shape[1]]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        donut_mask = (dist >= inner_r) & (dist <= outer_r) & (mask == 1)
        wafer[donut_mask] = 255
        
    elif pattern == 'near_full':
        # Almost all defective
        defect_mask = (np.random.random(wafer.shape) < 0.7) & (mask == 1)
        wafer[defect_mask] = 255
        
    else:  # mixed
        # Combination of patterns
        wafer = add_defects_center(wafer, mask)
        wafer = add_defects_random(wafer, mask)
    
    return wafer


# Map classes to their defect functions
DEFECT_FUNCTIONS = {
    'clean': add_defects_clean,
    'scratch': add_defects_scratch,
    'center_defect': add_defects_center,
    'edge_defect': add_defects_edge,
    'random_defect': add_defects_random,
    'short_bridge': add_defects_short_bridge,
    'open_circuit': add_defects_open_circuit,
    'other': add_defects_other
}


def generate_wafer_image(defect_class):
    """Generate a single wafer map image with the specified defect type."""
    wafer, mask = create_base_wafer(IMG_SIZE, WAFER_RADIUS)
    
    # Apply defect pattern
    defect_func = DEFECT_FUNCTIONS[defect_class]
    wafer = defect_func(wafer, mask)
    
    # Convert to PIL Image
    img = Image.fromarray(wafer, mode='L')
    return img


def generate_dataset():
    """Generate the complete synthetic dataset."""
    print("=" * 50)
    print("Synthetic Wafer Defect Dataset Generator")
    print("=" * 50)
    
    # Create output directories
    for cls in CLASSES:
        class_dir = OUTPUT_DIR / cls
        class_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    
    for cls in CLASSES:
        print(f"\nGenerating {IMAGES_PER_CLASS} images for class: {cls}")
        class_dir = OUTPUT_DIR / cls
        
        for i in range(IMAGES_PER_CLASS):
            img = generate_wafer_image(cls)
            
            # Add random augmentation variation
            if random.random() < 0.3:
                angle = random.randint(-10, 10)
                img = img.rotate(angle, fillcolor=0)
            
            # Save image
            filename = f"{cls}_{i:04d}.png"
            img.save(class_dir / filename)
            
            total_images += 1
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{IMAGES_PER_CLASS}...")
    
    print(f"\n{'=' * 50}")
    print(f"Dataset generation complete!")
    print(f"Total images: {total_images}")
    print(f"Classes: {len(CLASSES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
    
    # Print class distribution
    print("\nClass distribution:")
    for cls in CLASSES:
        class_dir = OUTPUT_DIR / cls
        count = len(list(class_dir.glob("*.png")))
        print(f"  {cls}: {count} images")


if __name__ == "__main__":
    generate_dataset()
