# SemiDiff - AI–Based Defect Classification System 🔍

**Version 1.1 | Developed by Team Ascension**

SemiDiff is a specialized AI system designed for detecting and classifying defects in semiconductor wafer die images. It uses a lightweight, optimized ONNX model for fast and accurate inference.

---

## 🎯 What's Included

```
To User/
├── inference.py          # Interactive Inference System
├── models/
│   └── model.onnx       # Trained ONNX model (0.95 MB)
├── sample_images/        # Test images
│   ├── test_scratch.png
│   ├── test_center.png
│   └── ...
├── requirements.txt      # Dependencies
└── README.md            # This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run SemiDiff

Start the interactive system:

```bash
python inference.py
```

You will see the **SemiDiff** interface:

```text
   _____                _ ____  _ ________
  / ___/___  ____ ___  (_) __ \(_) __/ __/
  \__ \/ _ \/ __ `__ \/ / / / / / /_/ /_  
 ...
          AI–Based Defect Classification System for Semiconductor WaferDie Images
                        made by Team Ascension | V1.1
```

### Step 3: Analyze Images

1.  The system will prompt you for an image path.
2.  Enter the path (e.g., `sample_images/test_scratch.png`).
3.  View the detailed classification results in a formatted table.
4.  The system automatically asks for the next image.
5.  Type `q` or `exit` to close the application.

---

## 📊 Model Details

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.62% |
| **F1-Score** | 95.55% |
| **Model Size** | 0.95 MB |
| **Input Size** | 64x64 grayscale |

---

## 🏷️ Defect Classes

The model classifies wafer images into 8 categories:

1.  **none** - Clean wafer
2.  **center** - Center defect
3.  **donut** - Donut-shaped defect
4.  **edge_loc** - Edge localized defect
5.  **edge_ring** - Edge ring defect
6.  **loc** - Local defect
7.  **scratch** - Scratch defect
8.  **random** - Random defect

---

## 💡 Notes

-   **Interactive CLI**: Now features a drag-and-drop friendly interface.
-   **Rich Visuals**: Results are presented in easy-to-read tables with confidence bars.
-   **Edge Ready**: Optimized for NXP i.MX RT devices.

---

## 🔧 Troubleshooting

**Missing Dependencies**
```bash
pip install -r requirements.txt
```
Make sure `rich` and `pyfiglet` are installed.

---

## 📧 Feedback

**Team Ascension**  

## Complete Dataset which Model is trained Links
'''
https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image
https://www.kaggle.com/datasets/husseinsalahyounis/wm-400k-wafer-map-single-and-mixed

'''
