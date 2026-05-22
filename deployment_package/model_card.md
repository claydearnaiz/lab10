# Model Card: ShuffleNetV2 Mulberry Classifier

## Source Model
- Model name: ShuffleNetV2
- Source framework: PyTorch / Torchvision
- Source weights: `shufflenet_best.pth`
- Dataset: Mulberry Classification
- Task type: Image classification

## Conversion Pathway
PyTorch `.pth` → ONNX → TensorFlow SavedModel → TensorFlow Lite `.tflite`

## Input
- Image size: 224 x 224
- Channels: 3
- Color order: RGB
- PyTorch training tensor format: NCHW
- TFLite inference tensor format: determined from interpreter input details

## Preprocessing
- Resize to 224 x 224
- Convert to float32
- Normalize with:
  - Mean: [0.485, 0.456, 0.406]
  - Standard deviation: [0.229, 0.224, 0.225]

## Output
- Output type: logits
- Number of classes: 5

## Class Order
0. Discolored
1. Healthy
2. Mold
3. Ripe
4. Unripe

## Verification Results
- Images checked: 25
- PyTorch accuracy: 1.0
- ONNX accuracy: 1.0
- ONNX agreement with PyTorch: 100.0%
- ONNX max absolute output error: 2.384185791015625e-06
- TFLite runtime status: runtime ok
- TFLite accuracy: 1.0
- TFLite agreement with PyTorch: 100.0
- TFLite max absolute output error: 2.1457672119140625e-06
- Used FlexOps: False

## Notes for Lab 10
The converted model must be re-tested on the actual Raspberry Pi hardware for runtime compatibility, latency, preprocessing correctness, and prediction consistency.
