#!/usr/bin/env python3
"""
Laboratory Experiment 10 Raspberry Pi deployment app.

Task: 5-class mulberry image classification.
Input: camera frame captured when the push button is pressed.
Output: one LED per class; the argmax class LED is lit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_LED_PINS = [5, 6, 13, 19, 26]
DEFAULT_BUTTON_PIN = 17


@dataclass
class Prediction:
    index: int
    label: str
    confidence: float
    probabilities: np.ndarray
    logits_or_scores: np.ndarray
    inference_ms: float


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def load_interpreter(model_path: Path, threads: int):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            try:
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter
            except ImportError:
                raise RuntimeError(
                    "Could not import tflite_runtime, ai_edge_litert, or tensorflow.lite. "
                    "On Raspberry Pi, install tflite-runtime. "
                    "On Windows/macOS/Linux PC, install ai-edge-litert for local testing."
                ) from exc

    interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    return interpreter


def tensor_hw_and_layout(input_details: dict) -> tuple[int, int, str]:
    shape = list(input_details["shape"])
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D image input tensor, got shape {shape}")

    # Most TFLite image models use NHWC: [1, height, width, channels].
    if shape[-1] == 3:
        return int(shape[1]), int(shape[2]), "NHWC"
    if shape[1] == 3:
        return int(shape[2]), int(shape[3]), "NCHW"
    raise ValueError(f"Could not infer image layout from input shape {shape}")


def preprocess_image(image: Image.Image, input_details: dict) -> np.ndarray:
    height, width, layout = tensor_hw_and_layout(input_details)
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))

    arr = np.expand_dims(arr, axis=0)
    dtype = input_details["dtype"]
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype)

    scale, zero_point = input_details.get("quantization", (0.0, 0))
    if not scale:
        raise ValueError(
            f"Model input is {dtype}, but no quantization scale was provided."
        )
    quantized = np.round(arr / scale + zero_point)
    info = np.iinfo(dtype)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


def dequantize_if_needed(values: np.ndarray, output_details: dict) -> np.ndarray:
    dtype = output_details["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(np.float32)
    scale, zero_point = output_details.get("quantization", (0.0, 0))
    if not scale:
        return values.astype(np.float32)
    return (values.astype(np.float32) - zero_point) * scale


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.squeeze(scores).astype(np.float32)
    if scores.ndim != 1:
        raise ValueError(f"Expected a 1D class vector after squeeze, got {scores.shape}")

    total = float(np.sum(scores))
    looks_like_probabilities = (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2)
    )
    return scores if looks_like_probabilities else softmax(scores)


def classify(interpreter, input_data: np.ndarray, labels: list[str]) -> Prediction:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    start = time.perf_counter()
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    inference_ms = (time.perf_counter() - start) * 1000.0

    raw_output = interpreter.get_tensor(output_details["index"])
    scores = np.squeeze(dequantize_if_needed(raw_output, output_details))
    probabilities = scores_to_probabilities(scores)

    if len(probabilities) != len(labels):
        raise ValueError(
            f"Model returned {len(probabilities)} classes, but labels.txt has {len(labels)}"
        )

    index = int(np.argmax(probabilities))
    return Prediction(
        index=index,
        label=labels[index],
        confidence=float(probabilities[index]),
        probabilities=probabilities,
        logits_or_scores=scores,
        inference_ms=inference_ms,
    )


def warm_up(interpreter) -> float:
    input_details = interpreter.get_input_details()[0]
    shape = [1 if int(dim) < 0 else int(dim) for dim in input_details["shape"]]
    dummy = np.zeros(shape, dtype=input_details["dtype"])
    start = time.perf_counter()
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()
    return (time.perf_counter() - start) * 1000.0


class ImageSource:
    def capture_rgb(self) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        return None


class StaticImageSource(ImageSource):
    def __init__(self, image_path: Path):
        self.image_path = image_path

    def capture_rgb(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


class PiCamera2Source(ImageSource):
    def __init__(self, size: tuple[int, int] = (640, 480)):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"format": "RGB888", "size": size}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1.0)

    def capture_rgb(self) -> Image.Image:
        frame = self.camera.capture_array()
        return Image.fromarray(frame, mode="RGB")

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()


class OpenCVCameraSource(ImageSource):
    def __init__(self, device: int):
        import cv2

        self.cv2 = cv2
        self.camera = cv2.VideoCapture(device)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open USB camera device {device}")
        time.sleep(0.5)

    def capture_rgb(self) -> Image.Image:
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the USB camera")
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")

    def close(self) -> None:
        self.camera.release()


class LedBoard:
    def __init__(self, pins: list[int], labels: list[str], dry_run: bool):
        self.pins = pins
        self.labels = labels
        self.dry_run = dry_run
        self.leds = []

        if len(pins) != len(labels):
            raise ValueError(
                f"Expected {len(labels)} LED pins, got {len(pins)}: {pins}"
            )

        if not dry_run:
            from gpiozero import LED

            self.leds = [LED(pin) for pin in pins]

    def all_off(self) -> None:
        if self.dry_run:
            return
        for led in self.leds:
            led.off()

    def show_class(self, index: int | None) -> None:
        self.all_off()
        if index is None:
            if self.dry_run:
                print("LED output: all LEDs OFF (below confidence threshold)")
            return

        if self.dry_run:
            print(
                f"LED output: GPIO {self.pins[index]} ON "
                f"for class {index} ({self.labels[index]})"
            )
            return
        self.leds[index].on()

    def close(self) -> None:
        self.all_off()
        for led in self.leds:
            led.close()


class CaptureButton:
    def __init__(self, pin: int, dry_run: bool):
        self.pin = pin
        self.dry_run = dry_run
        self.button = None
        if not dry_run:
            from gpiozero import Button

            self.button = Button(pin, pull_up=True, bounce_time=0.1)

    def wait_for_press(self) -> bool:
        if self.dry_run:
            try:
                value = input("Press Enter to capture, or type q + Enter to quit: ").strip()
            except EOFError:
                return False
            return value.lower() not in {"q", "quit", "exit"}

        self.button.wait_for_press()
        self.button.wait_for_release()
        return True

    def close(self) -> None:
        if self.button is not None:
            self.button.close()


def make_image_source(args) -> ImageSource:
    if args.camera == "image":
        if args.image is None:
            raise ValueError("--camera image requires --image")
        return StaticImageSource(args.image)

    if args.camera == "picamera2":
        return PiCamera2Source()

    if args.camera == "opencv":
        return OpenCVCameraSource(args.camera_device)

    if args.camera == "auto":
        try:
            return PiCamera2Source()
        except Exception as picamera_error:
            print(f"picamera2 unavailable: {picamera_error}")
            print("Trying USB/OpenCV camera instead...")
            return OpenCVCameraSource(args.camera_device)

    raise ValueError(f"Unknown camera backend: {args.camera}")


def top_k_text(prediction: Prediction, labels: list[str], k: int = 3) -> str:
    order = np.argsort(prediction.probabilities)[::-1][:k]
    return ", ".join(
        f"{labels[i]}={prediction.probabilities[i] * 100:.1f}%" for i in order
    )


def append_log(
    log_csv: Path,
    prediction: Prediction,
    response_ms: float,
    source: str,
) -> None:
    first_write = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if first_write:
            writer.writerow(
                [
                    "timestamp",
                    "source",
                    "predicted_index",
                    "predicted_label",
                    "confidence",
                    "inference_ms",
                    "response_ms",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                source,
                prediction.index,
                prediction.label,
                f"{prediction.confidence:.6f}",
                f"{prediction.inference_ms:.3f}",
                f"{response_ms:.3f}",
            ]
        )


def save_capture(image: Image.Image, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(path, quality=92)
    return path


def run_prediction_once(
    interpreter,
    labels: list[str],
    input_details: dict,
    image: Image.Image,
) -> Prediction:
    input_data = preprocess_image(image, input_details)
    return classify(interpreter, input_data, labels)


def run_check(args, interpreter, labels: list[str], input_details: dict) -> None:
    if args.image is None:
        raise ValueError("--check requires --image")
    image = Image.open(args.image).convert("RGB")
    prediction = run_prediction_once(interpreter, labels, input_details, image)
    print(
        f"Prediction: class {prediction.index} ({prediction.label}), "
        f"confidence {prediction.confidence * 100:.1f}%"
    )
    print(f"Top classes: {top_k_text(prediction, labels)}")
    print(f"Inference time: {prediction.inference_ms:.3f} ms")


def run_benchmark(args, interpreter, labels: list[str], input_details: dict) -> None:
    source = make_image_source(args)
    try:
        image = source.capture_rgb()
        input_data = preprocess_image(image, input_details)
        times = []

        for _ in range(args.benchmark):
            prediction = classify(interpreter, input_data, labels)
            times.append(prediction.inference_ms)

        times_arr = np.array(times, dtype=np.float32)
        print(f"Benchmark samples: {args.benchmark}")
        print(f"Mean inference time: {float(np.mean(times_arr)):.3f} ms/sample")
        print(f"Inference std. dev.: {float(np.std(times_arr, ddof=1)):.3f} ms")
        print(
            f"Last prediction: class {prediction.index} ({prediction.label}), "
            f"confidence {prediction.confidence * 100:.1f}%"
        )
    finally:
        source.close()


def run_button_loop(args, interpreter, labels: list[str], input_details: dict) -> None:
    source = make_image_source(args)
    leds = LedBoard(args.led_pins, labels, args.dry_run)
    button = CaptureButton(args.button_pin, args.dry_run)

    print("Ready.")
    print(f"Button BCM GPIO: {args.button_pin}")
    print(
        "LED mapping: "
        + ", ".join(
            f"class {idx} {label}->GPIO {pin}"
            for idx, (label, pin) in enumerate(zip(labels, args.led_pins))
        )
    )

    try:
        while True:
            if not button.wait_for_press():
                break

            response_start = time.perf_counter()
            image = source.capture_rgb()

            if args.save_captures:
                saved_path = save_capture(image, args.save_captures)
                print(f"Saved capture: {saved_path}")

            prediction = run_prediction_once(interpreter, labels, input_details, image)
            response_ms = (time.perf_counter() - response_start) * 1000.0

            led_index = (
                prediction.index if prediction.confidence >= args.confidence_threshold else None
            )
            leds.show_class(led_index)

            print(
                f"Prediction: class {prediction.index} ({prediction.label}), "
                f"confidence {prediction.confidence * 100:.1f}%, "
                f"inference {prediction.inference_ms:.3f} ms, "
                f"button-to-LED {response_ms:.3f} ms"
            )
            print(f"Top classes: {top_k_text(prediction, labels)}")

            if args.log_csv:
                append_log(args.log_csv, prediction, response_ms, args.camera)

            time.sleep(args.display_seconds)
            leds.all_off()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        leds.close()
        button.close()
        source.close()


def run_opencv_preview(args, interpreter, labels: list[str], input_details: dict) -> None:
    if args.camera != "opencv":
        raise ValueError("--preview currently works with --camera opencv only")

    import cv2

    camera = cv2.VideoCapture(args.camera_device)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open USB camera device {args.camera_device}")

    leds = LedBoard(args.led_pins, labels, args.dry_run)
    button = None
    if not args.dry_run:
        from gpiozero import Button

        button = Button(args.button_pin, pull_up=True, bounce_time=0.1)

    window_name = "Lab 10 USB Webcam Preview"
    status_text = "Press hardware button to classify, q to quit"
    led_off_at = 0.0
    button_was_pressed = False

    print("Preview ready.")
    print(f"Button BCM GPIO: {args.button_pin}")
    print("Hardware button = capture/classify")
    print("Keyboard backup: c or Space = capture/classify, q = quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the USB camera")

            if led_off_at and time.perf_counter() >= led_off_at:
                leds.all_off()
                led_off_at = 0.0

            display = frame.copy()
            cv2.putText(
                display,
                status_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break

            button_capture = False
            if button is not None:
                button_is_pressed = button.is_pressed
                button_capture = button_is_pressed and not button_was_pressed
                button_was_pressed = button_is_pressed

            keyboard_capture = key in {ord("c"), ord(" ")}
            if not button_capture and not keyboard_capture:
                continue

            response_start = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb, mode="RGB")

            if args.save_captures:
                saved_path = save_capture(image, args.save_captures)
                print(f"Saved capture: {saved_path}")

            prediction = run_prediction_once(interpreter, labels, input_details, image)
            response_ms = (time.perf_counter() - response_start) * 1000.0

            led_index = (
                prediction.index if prediction.confidence >= args.confidence_threshold else None
            )
            leds.show_class(led_index)
            led_off_at = time.perf_counter() + args.display_seconds

            status_text = (
                f"{prediction.label} {prediction.confidence * 100:.1f}% "
                f"({prediction.inference_ms:.1f} ms)"
            )
            print(
                f"Prediction: class {prediction.index} ({prediction.label}), "
                f"confidence {prediction.confidence * 100:.1f}%, "
                f"inference {prediction.inference_ms:.3f} ms, "
                f"capture-to-LED {response_ms:.3f} ms"
            )
            print(f"Top classes: {top_k_text(prediction, labels)}")

            if args.log_csv:
                append_log(args.log_csv, prediction, response_ms, args.camera)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if button is not None:
            button.close()
        leds.close()
        camera.release()
        cv2.destroyAllWindows()


def parse_led_pins(value: str) -> list[int]:
    try:
        pins = [int(pin.strip()) for pin in value.split(",") if pin.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("LED pins must be comma-separated integers") from exc
    return pins


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Button-triggered TFLite classification with class LEDs."
    )
    parser.add_argument("--model", type=existing_path, default=Path("model.tflite"))
    parser.add_argument("--labels", type=existing_path, default=Path("labels.txt"))
    parser.add_argument("--image", type=existing_path, help="Image for --check or --camera image")
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera2", "opencv", "image"],
        default="auto",
        help="Use picamera2 for a Pi Camera, opencv for USB webcam, or image for testing.",
    )
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--button-pin", type=int, default=DEFAULT_BUTTON_PIN)
    parser.add_argument(
        "--led-pins",
        type=parse_led_pins,
        default=DEFAULT_LED_PINS,
        help="Comma-separated BCM GPIO pins, one per class.",
    )
    parser.add_argument("--display-seconds", type=float, default=4.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Leave all LEDs off if the argmax confidence is below this value.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print GPIO actions instead of using pins.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a USB webcam preview window. Press c/Space to classify and q to quit.",
    )
    parser.add_argument("--check", action="store_true", help="Run one image sanity check and exit.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Run N warm benchmark inferences and exit.",
    )
    parser.add_argument("--save-captures", type=Path, help="Directory for captured frames.")
    parser.add_argument("--log-csv", type=Path, help="Append prediction timings to this CSV.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)

    labels = load_labels(args.labels)
    load_start = time.perf_counter()
    interpreter = load_interpreter(args.model, args.threads)
    load_allocate_ms = (time.perf_counter() - load_start) * 1000.0
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Loaded model: {args.model}")
    print(f"Model load + tensor allocation: {load_allocate_ms:.3f} ms")
    print(f"Input tensor: shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Output tensor: shape={output_details['shape']}, dtype={output_details['dtype']}")
    print(f"Labels: {labels}")

    if len(args.led_pins) != len(labels):
        raise ValueError(
            f"LED pin count ({len(args.led_pins)}) must match class count ({len(labels)})"
        )

    print("Warming up interpreter...")
    warmup_ms = warm_up(interpreter)
    print(f"Warm-up inference: {warmup_ms:.3f} ms")

    if args.check:
        run_check(args, interpreter, labels, input_details)
        return 0

    if args.benchmark > 0:
        run_benchmark(args, interpreter, labels, input_details)
        return 0

    if args.preview:
        run_opencv_preview(args, interpreter, labels, input_details)
        return 0

    run_button_loop(args, interpreter, labels, input_details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
