# Technical Approach Document
## Semiconductor Wafer Defect Detection using Deep Learning
### IESA DeepTech Hackathon 2026

---

## 1. Problem Understanding

### 1.1 Challenge
Develop an AI system for automated detection and classification of defects in semiconductor wafer images, optimized for edge deployment on NXP i.MX RT series devices.

### 1.2 Requirements
- Classify 8 categories of wafer defects
- Minimum 500 images (achieved: 800)
- Lightweight model for edge deployment
- High accuracy (target: >85%, achieved: 95.62%)

---

## 2. Approach

### 2.1 Dataset Strategy
- **Source**: Real WM811K semiconductor wafer images from Kaggle
- **No synthetic data used** - 100% real images
- **Classes**: 8 standard WM811K defect types
- **Total images**: 800 (100 per class)

### 2.2 Model Architecture
Custom lightweight CNN using depthwise separable convolutions:
- **Parameters**: 247,368 (~0.25M)
- **Size**: 0.95 MB (ONNX)
- **Input**: 64x64 grayscale
- **Output**: 8 classes

### 2.3 Training Approach
- **Framework**: PyTorch 2.6.0
- **Optimizer**: AdamW with cosine annealing LR
- **Augmentation**: Rotation, flip, affine, color jitter
- **Platform**: CPU (no GPU/cloud)

---

## 3. Dataset Plan

### 3.1 Data Sources
1. **WM811K Silicon Wafer Map Dataset Image** (Kaggle)
2. **Mixed-type Wafer Defect Datasets** (Kaggle)

### 3.2 Class Distribution

| Class | Description | Train | Val | Test | Total |
|-------|-------------|-------|-----|------|-------|
| none | Clean wafers | 70 | 15 | 15 | 100 |
| center | Center defect | 70 | 15 | 15 | 100 |
| donut | Donut pattern | 70 | 15 | 15 | 100 |
| edge_loc | Edge localized | 70 | 15 | 15 | 100 |
| edge_ring | Edge ring | 70 | 15 | 15 | 100 |
| loc | Local defect | 70 | 15 | 15 | 100 |
| scratch | Scratch | 70 | 15 | 15 | 100 |
| random | Random defects | 70 | 15 | 15 | 100 |
| **Total** | | 560 | 120 | 120 | 800 |

### 3.3 Preprocessing
- Resize to 64x64
- Convert to grayscale
- Normalize to [-1, 1]

---

## 4. Model Plan

### 4.1 Architecture Design

```
Input: 64x64x1 (grayscale)
    ↓
[Conv2D 32] → BatchNorm → ReLU
    ↓
[DepthwiseSeparable 64] → MaxPool2D
    ↓
[DepthwiseSeparable 128] → MaxPool2D
    ↓
[DepthwiseSeparable 256] → MaxPool2D
    ↓
[DepthwiseSeparable 512] → MaxPool2D
    ↓
[Global Average Pooling]
    ↓
[Dropout 0.3] → [FC 128] → ReLU
    ↓
[Dropout 0.15] → [FC 8]
    ↓
Output: 8-class probabilities
```

### 4.2 Why Depthwise Separable Convolutions?

Standard convolution: O(K² × C_in × C_out × H × W)
Depthwise separable: O(K² × C_in × H × W + C_in × C_out × H × W)

**Reduction**: ~8-9x fewer computations, enabling edge deployment.

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 30 (early stopped at 21) |
| Batch Size | 32 |
| Learning Rate | 0.001 → 0.00001 (cosine) |
| Weight Decay | 1e-4 |
| Early Stopping | Patience 10 |

---

## 5. Results

### 5.1 Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.62% |
| **Precision** | 95.80% |
| **Recall** | 95.62% |
| **F1-Score** | 95.55% |

### 5.2 Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| none | 100% | 100% | 100% |
| center | 100% | 90% | 95% |
| donut | 100% | 100% | 100% |
| edge_loc | 94% | 80% | 86% |
| edge_ring | 90% | 95% | 93% |
| loc | 91% | 100% | 95% |
| scratch | 91% | 100% | 95% |
| random | 100% | 100% | 100% |

### 5.3 Confusion Matrix

```
             none  center  donut  edge_loc  edge_ring  loc  scratch  random
none           20      0      0        0         0      0       0       0
center          0     18      0        0         0      2       0       0
donut           0      0     20        0         0      0       0       0
edge_loc        0      0      0       16         2      0       2       0
edge_ring       0      0      0        1        19      0       0       0
loc             0      0      0        0         0     20       0       0
scratch         0      0      0        0         0      0      20       0
random          0      0      0        0         0      0       0      20
```

---

## 6. Edge Deployment

### 6.1 Model Export

| Format | Size | Purpose |
|--------|------|---------|
| PyTorch (.pth) | 1.0 MB | Training checkpoint |
| ONNX (.onnx) | 0.95 MB | Cross-platform inference |
| TFLite (INT8) | ~250 KB | NXP i.MX RT deployment |

### 6.2 NXP eIQ Integration

1. Convert ONNX → TensorFlow Lite
2. Quantize to INT8 using representative dataset
3. Deploy using MCUXpresso IDE
4. Target: NXP i.MX RT series

### 6.3 Expected Edge Performance

| Metric | Value |
|--------|-------|
| Inference Time | <50 ms |
| RAM Usage | ~100 KB |
| Model Size (INT8) | ~250 KB |

---

## 7. Deliverables Summary

| Deliverable | File |
|-------------|------|
| Dataset (.zip) | `data/dataset.zip` |
| ONNX Model | `models/model.onnx` |
| Model Results | `results/model_results.md` |
| NXP eIQ Instructions | `docs/nxp_eiq_porting.md` |
| Complete Code | `src/` directory |
| GitHub Repository | (link to be added) |

---

## 8. Learnings

1. **Real Data Challenges**: WM811K pickle format incompatible with Python 3.14; solved by using pre-converted Kaggle datasets.

2. **Class Selection**: Used 8 standard WM811K classes instead of custom classes for reproducibility.

3. **Edge Optimization**: Depthwise separable convolutions crucial for achieving <1MB model size.

4. **Training Efficiency**: CPU-only training viable for small datasets (~3 minutes for 800 images).

---

## 9. Future Improvements

1. **Expand dataset** to 1000+ images per class
2. **Mixed-type defects** handling
3. **Knowledge distillation** for further compression
4. **Attention mechanisms** for better localization

---

**Team**: IESA DeepTech Hackathon 2026 Participant
**Date**: February 6, 2026
