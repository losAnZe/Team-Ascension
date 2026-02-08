# NXP eIQ Model Porting Instructions
## Semiconductor Wafer Defect Detection Model

---

## Model Information

| Property | Value |
|----------|-------|
| Source Format | ONNX |
| Target Format | TensorFlow Lite (INT8) |
| Model File | model.onnx |
| Size | 0.95 MB |
| Target Platform | NXP i.MX RT series |

---

## Step 1: Environment Setup

```bash
# Install NXP eIQ Toolkit
# Download from: https://www.nxp.com/eiq

# Install Python dependencies
pip install onnx tf2onnx tensorflow

# Verify ONNX model
python -c "import onnx; onnx.checker.check_model('models/model.onnx')"
```

---

## Step 2: Convert ONNX to TensorFlow Lite

```bash
# Option A: Using onnx-tf
pip install onnx-tf
onnx-tf convert -i models/model.onnx -o models/tf_model

# Convert to TFLite
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('models/tf_model')
tflite_model = converter.convert()
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)
"

# Option B: Direct conversion (if supported)
# python -m tf2onnx.convert --onnx models/model.onnx --output models/model.tflite
```

---

## Step 3: Quantize to INT8

```python
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Representative dataset for quantization calibration
def representative_dataset():
    data_dir = Path('data/submission/Train')
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            for img_path in list(class_dir.glob('*.png'))[:10]:
                img = Image.open(img_path).convert('L').resize((64, 64))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
                img_array = img_array.reshape(1, 64, 64, 1)
                yield [img_array]

# Convert with INT8 quantization
converter = tf.lite.TFLiteConverter.from_saved_model('models/tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()

with open('models/model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"INT8 model saved. Size: {len(tflite_quant_model) / 1024:.2f} KB")
```

---

## Step 4: Deploy to NXP i.MX RT

### 4.1 Import to MCUXpresso IDE

1. Open MCUXpresso IDE
2. Import eIQ example project for your target board
3. Replace the model file with `model_int8.tflite`
4. Update model configuration in `model_config.h`

### 4.2 Model Configuration

```c
// model_config.h
#define MODEL_INPUT_WIDTH    64
#define MODEL_INPUT_HEIGHT   64
#define MODEL_INPUT_CHANNELS 1
#define MODEL_NUM_CLASSES    8

// Class names
const char* CLASS_NAMES[] = {
    "none",      // Clean wafer
    "center",    // Center defect
    "donut",     // Donut pattern
    "edge_loc",  // Edge localized
    "edge_ring", // Edge ring
    "loc",       // Local defect
    "scratch",   // Scratch
    "random"     // Random defects
};
```

### 4.3 Inference Code

```c
// wafer_inference.c
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_int8.h"

// Tensor arena for model execution
constexpr int kTensorArenaSize = 100 * 1024;  // 100KB
uint8_t tensor_arena[kTensorArenaSize];

int classify_wafer(uint8_t* image_data) {
    // Get input tensor
    TfLiteTensor* input = interpreter->input(0);
    
    // Copy image data (already preprocessed to 64x64 grayscale)
    memcpy(input->data.uint8, image_data, 64 * 64);
    
    // Run inference
    interpreter->Invoke();
    
    // Get output
    TfLiteTensor* output = interpreter->output(0);
    
    // Find max class
    int max_class = 0;
    uint8_t max_score = 0;
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        if (output->data.uint8[i] > max_score) {
            max_score = output->data.uint8[i];
            max_class = i;
        }
    }
    
    return max_class;
}
```

---

## Step 5: Verify Deployment

```c
// test_model.c
void test_model() {
    printf("=== Wafer Defect Detection Model Test ===\n");
    printf("Model: model_int8.tflite\n");
    printf("Input: 64x64x1 grayscale\n");
    printf("Output: 8 classes\n");
    printf("Expected accuracy: >95%%\n");
    
    // Load test image
    uint8_t test_image[64 * 64];
    load_test_image("test_wafer.bin", test_image);
    
    // Classify
    int result = classify_wafer(test_image);
    printf("Classification result: %s\n", CLASS_NAMES[result]);
}
```

---

## Expected Performance on i.MX RT

| Metric | Value |
|--------|-------|
| Model Size (INT8) | ~250 KB |
| Inference Time | <50 ms |
| RAM Usage | ~100 KB |
| Accuracy | >90% (INT8) |

---

## Files for Deployment

```
models/
├── model.onnx          # Original ONNX model
├── model.tflite        # TFLite FP32 model
├── model_int8.tflite   # TFLite INT8 quantized
└── eiq_config.h        # NXP eIQ configuration
```

---

## References

- [NXP eIQ Documentation](https://www.nxp.com/eiq)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ONNX to TFLite Conversion](https://github.com/onnx/onnx-tensorflow)
