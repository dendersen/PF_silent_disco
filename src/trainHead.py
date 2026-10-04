from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from generate_datapoints import augment_image
from silent_disco import resolve_device, GenerateSettings

HEAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def prepare_head_dataset(source: Path, destination: Path, settings: GenerateSettings, max_side: int | None = None) -> tuple[int, int]:
    """Copy head images and augment pixels without changing YOLO boxes."""
    if destination.exists():
        shutil.rmtree(destination)
    image_destination = destination / "images"
    label_destination = destination / "labels"
    image_destination.mkdir(parents=True)
    label_destination.mkdir(parents=True)
    rng = np.random.default_rng(settings.seed)
    originals = 0
    generated = 0
    source_images = source / "images"
    source_labels = source / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise ValueError(f"Head dataset requires {source_images} and {source_labels}")
    image_paths = sorted(path for path in source_images.iterdir() if path.suffix.lower() in HEAD_IMAGE_EXTENSIONS)
    size_note = f"at up to {max_side}px" if max_side else "at original resolution"
    variant_count = settings.iterations if settings.enabled else 0
    progress = tqdm(total=len(image_paths) * (variant_count + 1), desc=f"Preparing head data {size_note}", unit="image")
    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.exists():
            continue
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width)) if max_side else 1.0
        processed_image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
        if not cv2.imwrite(str(image_destination / image_path.name), processed_image):
            raise ValueError(f"Could not write {image_destination / image_path.name}")
        shutil.copy2(label_path, label_destination / label_path.name)
        originals += 1
        progress.update(1)
        if not settings.enabled:
            continue
        for iteration in range(settings.iterations):
            augmented = augment_image(processed_image, rng, settings.noise_strength, 0, settings.smoke_strength)
            output_name = f"{image_path.stem}__aug{iteration:03d}{image_path.suffix.lower()}"
            output_path = image_destination / output_name
            if not cv2.imwrite(str(output_path), augmented):
                raise ValueError(f"Could not write {output_path}")
            shutil.copy2(label_path, label_destination / f"{Path(output_name).stem}.txt")
            generated += 1
            progress.update(1)
    progress.close()
    if originals == 0:
        raise ValueError(f"No labeled head images found under {source}")
    return originals, generated

def head_rocm_known_broken() -> bool:
    """Check if the current ROCm version is known to be broken for YOLO training."""
    version = torch.version.hip
    if version is None:
        return True  # Not running on ROCm, so considered broken
    known_broken_versions:list[str] = []
    for v in range(0,6):
        known_broken_versions.append(f"{v}.")
    known_broken_versions.append("7.0.51831")
    return any(version.startswith(broken) for broken in known_broken_versions)

def train_head_detector(source: Path, processed: Path, epochs: int, settings: GenerateSettings, requested_device: str, batch_size: int, image_size: int, model_path: Path, amp: bool, workers: int, is_subprocess: bool, reuse_processed: bool) -> None:
    """Train a one-class YOLO head detector on processed head data."""
    if requested_device == "rocm":
        if head_rocm_known_broken():
            if is_subprocess:
                raise RuntimeError(f"ROCm version is known to be broken for YOLO training, but this is supposed to be a known-good call!\nPlease check your ROCm version and try again.\nactuall: \"{torch.version.hip}\" should be: \"6.3.42134-a9a80e791\"")
            print("rocm version is known to be broken for YOLO training\ndownloading and/or running known good version now")
            subprocess.run(["python3", "-m", "venv", ".venv-rocm63"])
            pip_flags = ["--disable-pip-version-check", "--quiet", "--progress-bar", "on"]
            subprocess.run([".venv-rocm63/bin/python3", "-m", "pip", "install", *pip_flags, "--upgrade", "pip"])
            subprocess.run([".venv-rocm63/bin/python3", "-m", "pip", "install", *pip_flags, "-r", "src/requirements.txt"])
            subprocess.run([".venv-rocm63/bin/python3", "-m", "pip", "install", *pip_flags, "-r", "src/requirements-rocm_headTrain.txt"])
            subprocess.run([".venv-rocm63/bin/python3", "src/train_models.py", str(source.parent), "--kind", "head", "--processed-dataset", str(processed.parent), "--device", "rocm", "--epochs", str(epochs), "--head-batch-size", str(batch_size), "--head-image-size", str(image_size), "--isSubprocess", "True"])
            return
    if reuse_processed and (processed / "images").is_dir() and any((processed / "images").iterdir()):
        print(f"head: reusing existing processed data at {processed}", flush=True)
    else:
        originals, generated = prepare_head_dataset(source, processed, settings)
        print(f"Prepared head data: copied={originals} augmented={generated}", flush=True)
    yaml_path = processed / "data.yaml"
    yaml_path.write_text(
        f"path: {processed.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: head\n"
    )
    try:
        from ultralytics import YOLO
        from ultralytics.utils import nms as ultralytics_nms
    except ImportError as error:
        raise RuntimeError("Head training requires ultralytics; install src/requirements.txt") from error
    original_nms = ultralytics_nms.non_max_suppression

    def unlimited_nms(*args, **kwargs):
        kwargs["max_time_img"] = float("inf")
        return original_nms(*args, **kwargs)

    ultralytics_nms.non_max_suppression = unlimited_nms
    base_checkpoint = model_path
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_checkpoint))
    device = resolve_device(requested_device)
    print(f"Starting YOLO head training on {device}...", flush=True)
    model.train(# type: ignore
        data=str(yaml_path),
        epochs=max(1, epochs),
        imgsz=image_size,
        batch=max(1, batch_size),
        device=str(device),
        workers=workers,
        cache=False,
        amp=amp,
        project="models",
        name="head-detector",
        exist_ok=True,
    )
    print("saved models/head-detector/weights/best.pt", flush=True)
