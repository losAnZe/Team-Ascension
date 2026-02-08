"""
Inference script for wafer defect detection.
Loads ONNX model and classifies wafer images.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    exit(1)


# Class names
CLASSES = ['none', 'center', 'donut', 'edge_loc', 'edge_ring', 'loc', 'scratch', 'random']


def preprocess_image(image_path, size=64):
    """Load and preprocess image for inference."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Add batch and channel dimensions: (1, 1, H, W)
    img_array = img_array.reshape(1, 1, size, size)
    
    return img_array


def load_model(model_path):
    """Load ONNX model."""
    session = ort.InferenceSession(model_path)
    return session


def predict(session, image_array):
    """Run inference on preprocessed image."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    
    # Softmax
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Get prediction
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]
    
    return pred_class, confidence, probs


def main():
    parser = argparse.ArgumentParser(description='Wafer defect inference')
    parser.add_argument('image', type=str, help='Path to wafer image')
    parser.add_argument('--model', type=str, default='models/model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return
    
    print("=" * 50)
    print("Wafer Defect Detection Inference")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    session = load_model(args.model)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    img_array = preprocess_image(args.image)
    
    # Predict
    pred_class, confidence, probs = predict(session, img_array)
    
    # Results
    print("\n" + "-" * 50)
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence*100:.1f}%")
    
    # Top-k
    print(f"\nTop-{args.top_k} predictions:")
    top_indices = np.argsort(probs)[::-1][:args.top_k]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {CLASSES[idx]}: {probs[idx]*100:.1f}%")


if __name__ == "__main__":
    main()
