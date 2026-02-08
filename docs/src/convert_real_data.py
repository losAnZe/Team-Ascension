"""
Real WM811K Dataset Converter

Uses a low-level approach to extract wafer maps from the pickle file,
bypassing pandas compatibility issues with Python 3.14.
"""

import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import sys

# Monkey-patch to handle old pandas structures
class FakePandasIndex:
    def __init__(self, *args, **kwargs):
        pass
    def __reduce__(self):
        return (FakePandasIndex, ())

# Register fake modules
class FakeModule:
    Int64Index = FakePandasIndex
    Float64Index = FakePandasIndex
    Index = FakePandasIndex
    
    def __getattr__(self, name):
        return FakePandasIndex

sys.modules['pandas.indexes'] = FakeModule()
sys.modules['pandas.indexes.base'] = FakeModule()
sys.modules['pandas.core.indexes'] = FakeModule()
sys.modules['pandas.core.indexes.base'] = FakeModule()

# Configuration
INPUT_FILE = Path("data/kaggle/LSWMD.pkl")
OUTPUT_DIR = Path("data/processed")
IMG_SIZE = 64
TARGET_PER_CLASS = 150

# Class mapping
CLASS_MAP = {
    'none': 'clean',
    'Center': 'center_defect',
    'Edge-Loc': 'edge_defect',
    'Edge-Ring': 'edge_defect',
    'Loc': 'random_defect',
    'Random': 'random_defect',
    'Scratch': 'scratch',
    'Donut': 'other',
    'Near-full': 'other',
}

CLASSES = ['clean', 'scratch', 'center_defect', 'edge_defect', 
           'random_defect', 'short_bridge', 'open_circuit', 'other']


def wafer_to_image(wafer_map, size=64):
    """Convert wafer map array to grayscale image."""
    if wafer_map is None:
        return None
    
    wm = np.array(wafer_map, dtype=np.uint8)
    if wm.size == 0:
        return None
    
    img_array = np.zeros_like(wm, dtype=np.uint8)
    img_array[wm == 1] = 128  # Normal dies - gray
    img_array[wm == 2] = 255  # Defective dies - white
    
    img = Image.fromarray(img_array, mode='L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles old pandas objects."""
    
    def find_class(self, module, name):
        # Redirect old pandas modules
        if 'pandas' in module and 'index' in module.lower():
            return FakePandasIndex
        if name in ['Int64Index', 'Float64Index', 'Index']:
            return FakePandasIndex
        return super().find_class(module, name)


def load_pickle_raw():
    """Load pickle file with raw binary parsing if needed."""
    print(f"Loading {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
        print(f"Loaded successfully! Type: {type(data)}")
        return data
    except Exception as e:
        print(f"Custom unpickler failed: {e}")
        
        # Try with encoding
        try:
            with open(INPUT_FILE, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"Loaded with latin1 encoding! Type: {type(data)}")
            return data
        except Exception as e2:
            print(f"Latin1 encoding also failed: {e2}")
            return None


def process_dataframe(df):
    """Process the loaded dataframe and save images."""
    print("\nProcessing dataframe...")
    
    # Create output directories
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    counts = Counter()
    total = 0
    
    # Try to identify columns
    if hasattr(df, 'columns'):
        print(f"Columns: {list(df.columns)}")
    
    # Iterate through data
    for idx in range(min(len(df), 50000)):  # Limit iterations
        try:
            row = df.iloc[idx] if hasattr(df, 'iloc') else df[idx]
            
            # Get wafer map
            wm = row.get('waferMap') if hasattr(row, 'get') else row[0]
            if wm is None or (hasattr(wm, '__len__') and len(wm) == 0):
                continue
            
            # Get failure type
            ft = row.get('failureType') if hasattr(row, 'get') else row[1]
            if ft is None or len(ft) == 0:
                label = 'none'
            elif hasattr(ft, '__iter__') and len(ft) > 0:
                if hasattr(ft[0], '__iter__') and len(ft[0]) > 0:
                    label = str(ft[0][0])
                else:
                    label = str(ft[0])
            else:
                label = str(ft)
            
            # Map to our class
            cls = CLASS_MAP.get(label, 'other')
            
            # Check if we have enough of this class
            if counts[cls] >= TARGET_PER_CLASS:
                # Check if all classes are full
                if sum(1 for c in CLASSES if counts[c] >= TARGET_PER_CLASS) == len(CLASSES):
                    break
                continue
            
            # Convert to image
            img = wafer_to_image(wm, IMG_SIZE)
            if img is None:
                continue
            
            # Save
            filename = f"{cls}_{counts[cls]:04d}.png"
            img.save(OUTPUT_DIR / cls / filename)
            counts[cls] += 1
            total += 1
            
            if total % 100 == 0:
                print(f"Processed {total} images: {dict(counts)}")
                
        except Exception as e:
            continue
    
    print(f"\nDone! Total images: {total}")
    print("Class distribution:")
    for cls in CLASSES:
        print(f"  {cls}: {counts[cls]}")
    
    return counts


def main():
    print("=" * 60)
    print("WM811K Real Dataset Converter")
    print("=" * 60)
    
    # Clean existing processed data
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load pickle
    data = load_pickle_raw()
    
    if data is None:
        print("\nFailed to load pickle file.")
        return
    
    # Process
    counts = process_dataframe(data)
    
    print(f"\nImages saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
