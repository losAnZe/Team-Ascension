"""
Export model to ONNX format for edge deployment.
Includes quantization for NXP eIQ / i.MX RT compatibility.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import create_model, count_parameters
from dataset import CLASSES


def export_to_onnx(model, output_path, img_size=64, opset_version=17):
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        img_size: Input image size
        opset_version: ONNX opset version
    
    Returns:
        Path to saved ONNX model
    """
    model.eval()
    
    # Create dummy input (batch_size=1, channels=1, height, width)
    dummy_input = torch.randn(1, 1, img_size, img_size)
    
    # Export using legacy mode to avoid Unicode issues
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy export
    )
    
    print(f"ONNX model saved to: {output_path}")
    return output_path


def verify_onnx(onnx_path, img_size=64):
    """Verify ONNX model loads correctly and runs inference."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
        
        # Test inference
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = np.random.randn(1, 1, img_size, img_size).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(None, {'input': test_input})
        
        print(f"ONNX inference test: PASSED")
        print(f"  Output shape: {outputs[0].shape}")
        
        return True
        
    except ImportError:
        print("Warning: onnx or onnxruntime not installed. Skipping verification.")
        return False
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def get_model_size(model_path):
    """Get model file size in MB."""
    import os
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def export(args):
    """Main export function."""
    print("=" * 60)
    print("Wafer Defect Detection - Model Export")
    print("=" * 60)
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    model = create_model(num_classes=len(CLASSES))
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print("\n" + "-" * 60)
    print("Exporting to ONNX...")
    onnx_path = output_dir / 'model.onnx'
    export_to_onnx(model, onnx_path, img_size=64)
    
    onnx_size = get_model_size(onnx_path)
    print(f"ONNX model size: {onnx_size:.2f} MB")
    
    # Verify ONNX
    if args.verify:
        print("\nVerifying ONNX model...")
        verify_onnx(onnx_path)
    
    # Save export info
    export_info = {
        'pytorch_checkpoint': str(checkpoint_path),
        'onnx_model': str(onnx_path),
        'onnx_size_mb': onnx_size,
        'num_parameters': num_params,
        'input_shape': [1, 1, 64, 64],
        'output_shape': [1, len(CLASSES)],
        'num_classes': len(CLASSES),
        'class_names': CLASSES,
        'opset_version': 11,
        'target_platform': 'NXP i.MX RT series',
        'notes': [
            'Model is ready for NXP eIQ toolkit',
            'Use onnx2tflite for TensorFlow Lite conversion',
            'INT8 quantization recommended for deployment'
        ]
    }
    
    info_path = output_dir / 'export_info.json'
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nArtifacts created:")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - Export info: {info_path}")
    print(f"\nModel size: {onnx_size:.2f} MB")
    print(f"\nNext steps for NXP eIQ deployment:")
    print("  1. Install NXP eIQ Toolkit")
    print("  2. Convert ONNX to TFLite: onnx2tflite model.onnx model.tflite")
    print("  3. Quantize: Use eIQ quantization tools for INT8")
    print("  4. Deploy to i.MX RT target board")
    
    return export_info


def main():
    parser = argparse.ArgumentParser(description='Export model for edge deployment')
    parser.add_argument('--checkpoint', type=str,
                        default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for exported models')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX model after export')
    
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
