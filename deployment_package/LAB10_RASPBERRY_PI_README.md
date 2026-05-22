# Lab 10 Raspberry Pi Deployment Guide

This folder is ready to copy to the Raspberry Pi. It contains the trained TFLite model, labels, preprocessing notes, sample input, and the Raspberry Pi script for button-triggered camera inference with five class LEDs.

## Model Summary

- Task: mulberry image classification
- Raspberry Pi target: Raspberry Pi 4B 8 GB
- Model: `model.tflite`
- Input: RGB image, 224 x 224
- Preprocessing: resize, scale to `[0, 1]`, ImageNet normalization
- Output: 5 logits/classes
- Class order:
  - Class 0: Discolored
  - Class 1: Healthy
  - Class 2: Mold
  - Class 3: Ripe
  - Class 4: Unripe

The LED mapping for this classification task is simple: light only the LED for the highest-confidence class.

## Suggested GPIO Wiring

Use BCM GPIO numbering in the script.

| Role | Component | BCM GPIO | Physical Pin | Suggested LED Color |
|---|---:|---:|---:|---|
| Capture trigger | Push button | GPIO17 | Pin 11 | N/A |
| Class 0: Discolored | LED | GPIO5 | Pin 29 | Red |
| Class 1: Healthy | LED | GPIO6 | Pin 31 | Green |
| Class 2: Mold | LED | GPIO13 | Pin 33 | Blue or Yellow |
| Class 3: Ripe | LED | GPIO19 | Pin 35 | Yellow or Red |
| Class 4: Unripe | LED | GPIO26 | Pin 37 | White or Green |

Button wiring:

- One button leg to GPIO17, physical pin 11.
- The other button leg to any GND pin.
- No external resistor is needed because the script uses the Pi internal pull-up resistor.

LED wiring:

- GPIO pin -> 330 ohm resistor -> LED anode, the longer leg.
- LED cathode, the shorter leg -> GND.
- Each LED needs its own 330 ohm resistor.

## Transfer To The Pi

From this folder's parent directory on your laptop:

```bash
scp -r deployment_package <pi_user>@<pi_hostname_or_ip>:/home/<pi_user>/lab10_deployment
```

Example:

```bash
scp -r deployment_package pi@raspberrypi.local:/home/pi/lab10_deployment
```

Then SSH into the Pi:

```bash
ssh pi@raspberrypi.local
cd ~/lab10_deployment/deployment_package
```

## Raspberry Pi Setup

Use 64-bit Raspberry Pi OS. If you use the official Pi Camera, enable/test it first:

```bash
rpicam-hello
```

Install dependencies:

```bash
bash setup_pi.sh
source .venv/bin/activate
```

If `tflite-runtime` cannot be installed for your OS/Python version, keep the virtual environment active and install the TensorFlow Lite runtime wheel recommended by your instructor for aarch64 Raspberry Pi OS.

## Sanity Check Before Wiring

Run the included sample image through the model without GPIO:

```bash
python lab10_pi_app.py --check --image sample_input.jpg --dry-run
```

This confirms that `model.tflite`, `labels.txt`, and the preprocessing code load correctly on the Pi.

## Optional PC Test Before Using The Pi

You can test the model and preprocessing on your PC first. This does not test the real Raspberry Pi GPIO LEDs or Pi camera, but it confirms that the TFLite model runs and that the script maps the predicted class to the correct LED in dry-run mode.

From `C:\Users\clayd\OneDrive\Documents\Lab 10` in PowerShell:

```powershell
py -3 -m venv .pcvenv
.\.pcvenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy Pillow ai-edge-litert
cd .\deployment_package
python .\lab10_pi_app.py --check --image .\sample_input.jpg --dry-run
python .\lab10_pi_app.py --camera image --image .\sample_input.jpg --dry-run
```

For a PC webcam dry-run, also install OpenCV:

```powershell
python -m pip install opencv-python
python .\lab10_pi_app.py --camera opencv --camera-device 0 --dry-run
```

Press Enter to simulate the hardware button. Type `q` and press Enter to quit.

If PowerShell blocks activation, run this once for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If `ai-edge-litert` fails to download, retry later or use another internet connection. On Windows, `tflite-runtime` usually is not available through pip, so `ai-edge-litert` is the easier PC test runtime.

## Run With Real Hardware

For a Pi Camera Module:

```bash
python lab10_pi_app.py --camera picamera2
```

For a USB webcam:

```bash
python lab10_pi_app.py --camera opencv --camera-device 0
```

For a USB webcam with a live preview window:

```bash
python lab10_pi_app.py --camera opencv --camera-device 0 --preview
```

In preview mode, press the hardware button to classify the current frame. You can also press `c` or Space as a keyboard backup, and press `q` to quit.

The defaults are:

```text
Button: GPIO17
LEDs:   GPIO5, GPIO6, GPIO13, GPIO19, GPIO26
```

To use different pins:

```bash
python lab10_pi_app.py --camera picamera2 --button-pin 17 --led-pins 5,6,13,19,26
```

## Capture Logs For The Report

Save captured images and timing logs:

```bash
python lab10_pi_app.py --camera picamera2 --save-captures captures --log-csv lab10_results.csv
```

Benchmark at least 30 warm inferences:

```bash
python lab10_pi_app.py --camera picamera2 --benchmark 30
```

For a model-only benchmark using the included sample image:

```bash
python lab10_pi_app.py --camera image --image sample_input.jpg --benchmark 30 --dry-run
```

Use these Lab 9 development-machine baseline values in the report comparison:

| Metric | Lab 9 TFLite Baseline |
|---|---:|
| Model file size | 4.816 MB |
| Accuracy | 1.0 |
| Mean inference latency | 7.196 ms/sample |
| Max absolute output error vs. PyTorch | 2.145767e-06 |
| Prediction agreement with PyTorch | 100.0% |

## Report Checklist

- Photograph of Raspberry Pi, breadboard, button, LEDs, and camera.
- GPIO assignment table using BCM and physical pin numbers.
- Screenshot or copied console output from `--check`.
- Six end-to-end tests: one per class plus one challenging case.
- Short 15 to 30 second video showing camera aim, button press, and LED response.
- Benchmark table: mean inference time, std. dev., button-to-LED response time, cold-start time, peak RAM, and accuracy retention.
- Discussion of at least three limitations or failure modes.
