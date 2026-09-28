"""Train the two small models used by silent_disco.py.

Expected dataset layout:
  dataset/presence/negative/*.jpg
  dataset/presence/headset/*.jpg
    dataset/color/green/*.jpg, blue/*.jpg, red/*.jpg
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generate_datapoints import augment_dataset, augment_image
from silent_disco import COLORS, SmallConvNet, resolve_device


HEAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class GenerateSettings:
    enabled: bool = True
    iterations: int = 4
    noise_strength: float = 0.15
    max_shift: int = 10
    smoke_strength: float = 0.15
    seed: int = 20260926
    delete_dataset_after_training: bool = True


def load_generate_settings(path: Path) -> GenerateSettings:
    settings = GenerateSettings()
    if not path.exists():
        return settings
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section("generate"):
        return settings
    values = parser["generate"]
    result = GenerateSettings(
        enabled=values.getboolean("enabled", fallback=settings.enabled),
        iterations=values.getint("iterations", fallback=settings.iterations),
        noise_strength=values.getfloat("noise_strength", fallback=settings.noise_strength),
        max_shift=values.getint("max_shift", fallback=settings.max_shift),
        smoke_strength=values.getfloat("smoke_strength", fallback=settings.smoke_strength),
        seed=values.getint("seed", fallback=settings.seed),
        delete_dataset_after_training=values.getboolean("delete_dataset_after_training", fallback=settings.delete_dataset_after_training),
    )
    if result.iterations < 0 or result.max_shift < 0 or not 0 <= result.noise_strength <= 1 or not 0 <= result.smoke_strength <= 1:
        raise ValueError("generate.settings requires non-negative iterations/max_shift and strengths between 0 and 1")
    return result


class CropDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, labels: tuple[str, ...]) -> None:
        self.items = [(path, index) for index, label in enumerate(labels) for path in sorted((root / label).glob("*"))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read {path}")
        image = cv2.cvtColor(cv2.resize(image, (255, 255)), cv2.COLOR_BGR2RGB)
        array = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(array.transpose(2, 0, 1)), label


def prepare_head_dataset(source: Path, destination: Path, settings: GenerateSettings) -> tuple[int, int]:
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
    for image_path in sorted(source_images.iterdir()):
        if image_path.suffix.lower() not in HEAD_IMAGE_EXTENSIONS:
            continue
        label_path = source_labels / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.exists():
            continue
        shutil.copy2(image_path, image_destination / image_path.name)
        shutil.copy2(label_path, label_destination / label_path.name)
        originals += 1
        if not settings.enabled:
            continue
        for iteration in range(settings.iterations):
            augmented = augment_image(image, rng, settings.noise_strength, 0, settings.smoke_strength)
            output_name = f"{image_path.stem}__aug{iteration:03d}{image_path.suffix.lower()}"
            output_path = image_destination / output_name
            if not cv2.imwrite(str(output_path), augmented):
                raise ValueError(f"Could not write {output_path}")
            shutil.copy2(label_path, label_destination / f"{Path(output_name).stem}.txt")
            generated += 1
    if originals == 0:
        raise ValueError(f"No labeled head images found under {source}")
    return originals, generated


def train_head_detector(source: Path, processed: Path, epochs: int, settings: GenerateSettings) -> None:
    """Train a one-class YOLO head detector on processed head data."""
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
    except ImportError as error:
        raise RuntimeError("Head training requires ultralytics; install src/requirements.txt") from error
    model = YOLO("yolo11n.pt")
    model.train(data=str(yaml_path), epochs=min(15, max(1, epochs)), imgsz=1280, device="cpu", project="models", name="head-detector", exist_ok=True)
    print("saved models/head-detector/weights/best.pt", flush=True)


def wait_for_enter(stop_event: threading.Event) -> None:
    try:
        input()
        stop_event.set()
    except EOFError:
        return


def train(root: Path, labels: tuple[str, ...], output: Path, epochs: int, batch_size: int, target_loss: float | None, max_time: float | None, requested_device: str) -> None:
    dataset = CropDataset(root, labels)
    if not dataset:
        raise ValueError(f"No training crops found under {root}")
    device = resolve_device(requested_device)
    model = SmallConvNet(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    stop_event = threading.Event()
    threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True).start()
    started = time.monotonic()
    epoch = 0
    stop_reason = "user"
    print(f"Training {len(dataset)} processed crops")
    print("Press Enter to stop training.")
    while True:
        if stop_event.is_set():
            stop_reason = "user"
            break
        if max_time is not None and time.monotonic() - started >= max_time:
            stop_reason = "time"
            break
        if epochs > 0 and epoch >= epochs:
            stop_reason = "epoch"
            break
        epoch += 1
        model.train()
        total_loss = 0.0
        samples_seen = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs or '∞'}", unit="batch", leave=False)
        for images, targets in progress:
            if stop_event.is_set() or (max_time is not None and time.monotonic() - started >= max_time):
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            samples_seen += len(images)
            total_loss += loss.item() * len(images)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        progress.close()
        average_loss = total_loss / max(1, samples_seen)
        print(f"epoch={epoch} loss={average_loss:.4f} elapsed={time.monotonic() - started:.1f}s")
        if stop_event.is_set():
            stop_reason = "user"
            break
        if max_time is not None and time.monotonic() - started >= max_time:
            stop_reason = "time"
            break
        if target_loss is not None and average_loss <= target_loss:
            stop_reason = "loss"
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output)
    print(f"saved {output} after {epoch} epochs (stop={stop_reason})")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--kind", choices=("head", "presence", "color"), help="Train one model; omit to train presence and color.")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Model file for one kind, or output directory when training both.")
    parser.add_argument("--settings", type=Path, default=Path("generate.settings"), help="Augmentation settings file.")
    parser.add_argument("--processed-dataset", type=Path, default=Path("dataset_processed"), help="Temporary on-disk augmented dataset.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm", "xpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs; 0 means no epoch limit.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-loss", type=float, help="Stop when the epoch average loss reaches this value.")
    parser.add_argument("--max-time", type=float, help="Stop after this many seconds.")
    args = parser.parse_args()
    generate = load_generate_settings(args.settings)
    kinds = (args.kind,) if args.kind else ("presence", "color")
    if len(kinds) > 1 and args.output.suffix:
        parser.error("--output must be a directory when --kind is omitted")
    if kinds == ("head",):
        train_head_detector(args.dataset / "heads", args.processed_dataset / "head", args.epochs, generate)
        if generate.delete_dataset_after_training and args.processed_dataset.is_dir():
            shutil.rmtree(args.processed_dataset, ignore_errors=True)
        return

    print(f"Generating processed dataset at {args.processed_dataset}...", flush=True)
    for kind in kinds:
        source_labels = ("negative", "headset") if kind == "presence" else COLORS
        processed_kind = args.processed_dataset / kind
        if processed_kind.exists() and any(processed_kind.iterdir()):
            print(f"{kind}: reusing existing processed data at {processed_kind}", flush=True)
        else:
            originals, generated = augment_dataset(args.dataset / kind, processed_kind, generate.iterations if generate.enabled else 0, generate.noise_strength, generate.max_shift, generate.smoke_strength, generate.seed, True, source_labels)
            print(f"{kind}: copied={originals} augmented={generated}", flush=True)
    for kind in kinds:
        print(f"Training {kind} model...")
        labels = ("negative", "headset") if kind == "presence" else COLORS
        output = args.output / f"{kind}.pt" if args.output.is_dir() or not args.output.suffix else args.output
        train(args.processed_dataset / kind, labels, output, args.epochs, args.batch_size, args.target_loss, args.max_time, args.device)
        print(f"Finished training {kind} model.")
    if generate.delete_dataset_after_training and args.processed_dataset.is_dir():
        print("Deleting temporary dataset after training...")
        shutil.rmtree(args.processed_dataset, ignore_errors=True)
        open(os.path.join(args.processed_dataset, ".gitkeep"), "a").close()
        print(f"deleted temporary dataset {args.processed_dataset}")


def restore_terminal_echo() -> None:
    if sys.stdin.isatty():
        subprocess.run(("stty", "echo"), stdin=sys.stdin, check=False)


def main() -> None:
    try:
        _main()
    finally:
        restore_terminal_echo()


if __name__ == "__main__":
    main()