# Laboratory Exercise 10 Report Template

Replace the bracketed placeholders after testing on the Raspberry Pi.

## A. System Description

The deployed system used a Raspberry Pi 4B with 8 GB RAM running 64-bit Raspberry Pi OS. The model deployed was `model.tflite`, a ShuffleNetV2 TensorFlow Lite classifier trained for five mulberry classes: Discolored, Healthy, Mold, Ripe, and Unripe. The camera used was [Pi Camera Module / USB webcam]. A push button was connected to GPIO17 and configured with the Raspberry Pi internal pull-up resistor. Five LEDs were connected as class indicators, with one GPIO output per class.

| Role | Component | BCM GPIO | Physical Pin | LED Color |
|---|---:|---:|---:|---|
| Capture trigger | Push button | GPIO17 | Pin 11 | N/A |
| Class 0: Discolored | LED | GPIO5 | Pin 29 | [color] |
| Class 1: Healthy | LED | GPIO6 | Pin 31 | [color] |
| Class 2: Mold | LED | GPIO13 | Pin 33 | [color] |
| Class 3: Ripe | LED | GPIO19 | Pin 35 | [color] |
| Class 4: Unripe | LED | GPIO26 | Pin 37 | [color] |

The output rule was classification argmax: after each button press, the captured frame was resized to 224 x 224, converted to RGB, normalized using ImageNet mean and standard deviation, and passed to the TFLite interpreter. The LED corresponding to the highest-confidence class was lit for 4 seconds. No confidence threshold was used unless stated otherwise.

On-device sanity-check output:

```text
[paste output from: python lab10_pi_app.py --check --image sample_input.jpg --dry-run]
```

## B. End-To-End Behavior

| # | Input Scenario | Expected LED Behavior | Actual LED Behavior | Pass / Fail |
|---:|---|---|---|---|
| 1 | Discolored sample | Discolored LED turns on | [result] | [Pass/Fail] |
| 2 | Healthy sample | Healthy LED turns on | [result] | [Pass/Fail] |
| 3 | Mold sample | Mold LED turns on | [result] | [Pass/Fail] |
| 4 | Ripe sample | Ripe LED turns on | [result] | [Pass/Fail] |
| 5 | Unripe sample | Unripe LED turns on | [result] | [Pass/Fail] |
| 6 | Challenging/ambiguous sample | [expected behavior] | [result] | [Pass/Fail] |

During testing, the system was most reliable for [classes]. The most common errors occurred for [classes or conditions], especially when [lighting/focus/background/ambiguous ripeness]. These errors are likely caused by [model limitation / camera condition / preprocessing mismatch / LED wiring issue]. The button response was [reliable / sometimes double-counted / sometimes missed], and the LED mapping was verified by checking that each predicted class matched the expected GPIO pin.

## C. Performance And Accuracy

| Metric | Dev Machine From Lab 9 | Raspberry Pi This Lab |
|---|---:|---:|
| Mean inference time (ms/sample) | 7.196 | [mean] |
| Inference time std. dev. (ms) | [Lab 9 std. if available] | [std] |
| End-to-end response time, button to LED (ms) | N/A | [mean response] |
| Cold-start/load + warm-up time (ms) | N/A | [startup timing] |
| Peak RAM during inference (MB) | [if available] | [peak RAM] |
| Primary task metric, accuracy | 1.0 | [accuracy on test cases] |

The Raspberry Pi inference latency was [slower/faster] than the development-machine latency. This is expected because the Raspberry Pi uses an ARM CPU and does not use the same desktop CPU/GPU resources available during development. Since this model was not quantized and the conversion comparison showed 100% prediction agreement with PyTorch, any major accuracy drop on the Pi would most likely come from camera image quality or preprocessing mismatch rather than the TFLite model itself.

## D. Failure Modes And Limitations

1. Lighting changes can reduce accuracy because the camera may capture colors and shadows differently from the training images.
2. Similar visual stages, especially Ripe versus Unripe or Mold versus Discolored, may be difficult for the model under ambiguous conditions.
3. The LED interface only shows the top predicted class, so it does not communicate uncertainty unless the console confidence score is checked.
4. Button debounce and wiring quality can affect whether one physical press is detected exactly once.
5. Camera focus, motion blur, and distance from the mulberry can affect the captured frame before inference.

## Questions

### 1. Why is `tflite_runtime` preferred over the full `tensorflow` package on a Raspberry Pi?

`tflite_runtime` is preferred because it contains only the TensorFlow Lite interpreter and the kernels needed for inference. It is much smaller than the full `tensorflow` package, has fewer dependencies, and is easier to install on ARM-based Raspberry Pi OS. The full TensorFlow package includes training tools and extra APIs that are unnecessary for this deployment. Using the lighter runtime also reduces storage use, installation time, and memory overhead while still supporting efficient CPU inference.

### 2. Describe how you wired one LED and the push-button.

Each LED was wired from a Raspberry Pi GPIO pin through a 330 ohm current-limiting resistor to the LED anode, while the LED cathode was connected to ground. The resistor limits current so the GPIO pin and LED are not damaged. The push button was wired between GPIO17 and ground. No external resistor was required for the button because the software enables the Raspberry Pi internal pull-up resistor, so the input normally reads HIGH and changes to LOW when the button is pressed.

### 3. Explain how LED-output mapping differs across classification, object detection, and segmentation.

For classification, the model outputs one vector of class scores, so the system lights the LED for the class with the highest probability. For object detection, the model can output several detected objects, so the system can light multiple class LEDs after applying confidence thresholding and non-maximum suppression. For segmentation, the model outputs class predictions per pixel, so LED behavior can be based on the dominant pixel class or every class above a pixel-area threshold. This project uses the classification rule: only the argmax class LED is lit.

### 4. Why is the first inference slower, and how was cold-start latency handled?

The first inference is usually slower because the interpreter must prepare tensors, select kernels, and initialize runtime resources. To avoid making the user experience that delay after the first button press, the script loads the model, allocates tensors, and performs one dummy warm-up inference during startup. After that warm-up, button-triggered inference uses the steady-state runtime path.

### 5. Compare the on-Pi latency to Lab 9 development-machine latency.

The Lab 9 TFLite baseline mean inference latency was 7.196 ms/sample on the development machine. The Raspberry Pi latency was [measured value] ms/sample. The difference is mainly due to CPU architecture, clock speed, memory bandwidth, thermal limits, and the lack of desktop-class acceleration on the Raspberry Pi. The design choice with the largest effect on latency was keeping the model and input resolution modest: this deployment uses a 224 x 224 input and a lightweight ShuffleNetV2 architecture.

## Conclusion

This laboratory exercise deployed a five-class mulberry TensorFlow Lite classifier on a Raspberry Pi 4B with 8 GB RAM. The hardware interface used one push button on GPIO17 to trigger image capture and five class LEDs connected to GPIO5, GPIO6, GPIO13, GPIO19, and GPIO26. The model handled a classification task, so the LED-output mapping used the argmax class rule: after each captured frame was preprocessed and passed through the model, the LED for the highest-confidence class was lit. During testing, the system correctly identified [reliable classes] and had difficulty with [failure cases]. The most important observed failure modes were [failure mode 1], [failure mode 2], and [failure mode 3]. The measured Raspberry Pi mean inference time was [mean] ms/sample, with an end-to-end button-to-LED response time of [response] ms. Compared with the Lab 9 development-machine latency of 7.196 ms/sample, the Pi was [comparison] because inference ran on a smaller ARM CPU. Accuracy retention was [result], showing that the converted TFLite model [did/did not] preserve the Lab 9 behavior on actual hardware. Overall, the Lab 6 to Lab 10 workflow demonstrated the complete process of collecting data, training a model, converting it for edge deployment, and connecting the model output to a physical interface.
