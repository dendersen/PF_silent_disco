"""Generate augmented training crops before model training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def crop_with_shift(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Shift the crop center while preserving the original crop dimensions."""
    if shift_x == 0 and shift_y == 0:
        return image.copy()
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(image, height, height, width, width, cv2.BORDER_REFLECT_101)
    center_x = width + width // 2 + shift_x
    center_y = height + height // 2 + shift_y
    return padded[center_y - height // 2:center_y - height // 2 + height, center_x - width // 2:center_x - width // 2 + width]


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_strength: float,
    max_shift: int,
    smoke_strength: float,
) -> np.ndarray:
    """Apply bounded sensor noise, crop-center movement, and smoky lighting."""
    height, width = image.shape[:2]
    shift_x = int(rng.integers(-max_shift, max_shift + 1))
    shift_y = int(rng.integers(-max_shift, max_shift + 1))
    augmented = crop_with_shift(image, shift_x, shift_y)

    if noise_strength:
        sigma = 30.0 * noise_strength
        noise = rng.normal(0.0, sigma, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if smoke_strength:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 - smoke_strength
        hsv[:, :, 2] = np.minimum(255.0, hsv[:, :, 2] + 40.0 * smoke_strength)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if augmented.shape[:2] != (height, width):
        raise RuntimeError("Augmentation changed the crop dimensions")
    return augmented


def copy_originals(source: Path, destination: Path) -> int:
    copied = 0
    for label_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        target_dir = destination / label_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            shutil.copy2(path, target_dir / path.name)
            copied += 1
    return copied


def augment_dataset(
    source: Path,
    destination: Path,
    iterations: int,
    noise_strength: float,
    max_shift: int,
    smoke_strength: float,
    seed: int,
    include_originals: bool,
    include_labels: tuple[str, ...] | None = None,
) -> tuple[int, int]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination is not empty: {destination}")
    if iterations < 0 or max_shift < 0 or not 0 <= noise_strength <= 1 or not 0 <= smoke_strength <= 1:
        raise ValueError("iterations and max-shift must be non-negative; strengths must be between 0 and 1")

    rng = np.random.default_rng(seed)
    originals = 0
    generated = 0
    for label_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if include_labels is not None and label_dir.name not in include_labels:
            continue
        target_dir = destination / label_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read {path}")
            if include_originals:
                shutil.copy2(path, target_dir / path.name)
                originals += 1
            for iteration in range(iterations):
                augmented = augment_image(image, rng, noise_strength, max_shift, smoke_strength)
                output = target_dir / f"{path.stem}__aug{iteration:03d}{path.suffix.lower()}"
                if not cv2.imwrite(str(output), augmented):
                    raise ValueError(f"Could not write {output}")
                generated += 1
    return originals, generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented crops for presence or color training.")
    parser.add_argument("source", type=Path, help="Dataset task directory, such as dataset/color")
    parser.add_argument("destination", type=Path, help="Empty output task directory, such as dataset_processed/color")
    parser.add_argument("--iterations", type=int, default=4, help="Augmented crops generated per source crop.")
    parser.add_argument("--noise-strength", type=float, default=0.15, help="Gaussian noise strength from 0 to 1.")
    parser.add_argument("--max-shift", type=int, default=10, help="Maximum horizontal and vertical center shift in pixels.")
    parser.add_argument("--smoke-strength", type=float, default=0.15, help="Desaturation and brightness lift strength from 0 to 1.")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--no-originals", action="store_true", help="Write only augmented crops.")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"source directory does not exist: {args.source}")
    originals, generated = augment_dataset(args.source, args.destination, args.iterations, args.noise_strength, args.max_shift, args.smoke_strength, args.seed, not args.no_originals)
    print(f"copied_originals={originals} generated_augmented={generated} destination={args.destination}")


if __name__ == "__main__":
    main()