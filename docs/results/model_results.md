# Model Results - Semiconductor Wafer Defect Detection
## IESA DeepTech Hackathon 2026

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.62% |
| **Precision** | 95.80% |
| **Recall** | 95.62% |
| **F1-Score** | 95.55% |
| **Model Size** | 0.95 MB (ONNX) |
| **Parameters** | 247,368 |

---

## Algorithm Details

| Specification | Value |
|---------------|-------|
| **Algorithm** | Custom Lightweight CNN with Depthwise Separable Convolutions |
| **Framework** | PyTorch 2.6.0 |
| **Input Size** | 64x64 grayscale |
| **Output Classes** | 8 |
| **Training Platform** | CPU (Intel) |
| **Inference Platform** | CPU / NXP i.MX RT (TFLite) |
| **GPU Used** | No |
| **Cloud Used** | No (Local training) |

---

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| none (clean) | 100.0% | 100.0% | 100.0% | 20 |
| center | 100.0% | 90.0% | 95.0% | 20 |
| donut | 100.0% | 100.0% | 100.0% | 20 |
| edge_loc | 94.0% | 80.0% | 86.0% | 20 |
| edge_ring | 90.0% | 95.0% | 93.0% | 20 |
| loc | 91.0% | 100.0% | 95.0% | 20 |
| scratch | 91.0% | 100.0% | 95.0% | 20 |
| random | 100.0% | 100.0% | 100.0% | 20 |

---

## Confusion Matrix

```
Predicted →
Actual ↓      none  center  donut  edge_loc  edge_ring  loc  scratch  random

none           20      0      0        0         0       0       0       0
center          0     18      0        0         0       2       0       0
donut           0      0     20        0         0       0       0       0
edge_loc        0      0      0       16         2       0       2       0
edge_ring       0      0      0        1        19       0       0       0
loc             0      0      0        0         0      20       0       0
scratch         0      0      0        0         0       0      20       0
random          0      0      0        0         0       0       0      20
```

---

## Dataset Information

| Split | Images | Classes |
|-------|--------|---------|
| Train | 560 | 8 |
| Validation | 120 | 8 |
| Test | 120 | 8 |
| **Total** | **800** | 8 |

### Classes (Standard WM811K)
1. none (clean) - Normal wafers
2. center - Center defect
3. donut - Donut pattern
4. edge_loc - Edge localized defect
5. edge_ring - Edge ring defect
6. loc - Local defect
7. scratch - Scratch defect
8. random - Random defects

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 30 (early stopped at 21) |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 0.001 (cosine annealing) |
| Weight Decay | 1e-4 |
| Training Time | 3.3 minutes |
| Best Validation Accuracy | 93.8% (epoch 21) |

---

## Model Architecture

```
Input: 64x64x1 (grayscale)
    ↓
[Conv2D 32] → BatchNorm → ReLU
    ↓
[DepthwiseSeparable 64] → MaxPool
    ↓
[DepthwiseSeparable 128] → MaxPool
    ↓
[DepthwiseSeparable 256] → MaxPool
    ↓
[DepthwiseSeparable 512] → MaxPool
    ↓
[Global Average Pooling]
    ↓
[Dropout 0.3] → [FC 128] → ReLU
    ↓
[Dropout 0.15] → [FC 8]
    ↓
Output: 8-class probabilities
```

---

## Files Submitted

- `model.onnx` - Trained model in ONNX format
- `dataset.zip` - Train/Validation/Test images
- `model_results.md` - This file
- `approach.pdf` - Technical documentation
- GitHub repository with complete code
