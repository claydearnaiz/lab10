#!/usr/bin/env bash
set -euo pipefail

echo "Installing Lab 10 Raspberry Pi dependencies..."
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  python3-numpy \
  python3-pil \
  python3-gpiozero \
  python3-lgpio \
  python3-picamera2 \
  python3-opencv

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if ! python -m pip install tflite-runtime; then
  echo
  echo "Could not install tflite-runtime automatically."
  echo "Install the aarch64 TensorFlow Lite runtime wheel required by your Pi OS/Python version,"
  echo "then run: python lab10_pi_app.py --check --image sample_input.jpg --dry-run"
  exit 1
fi

echo
echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Sanity check: python lab10_pi_app.py --check --image sample_input.jpg --dry-run"
