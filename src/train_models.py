"""Train the two small models used by silent_disco.py.

Expected dataset layout:
  dataset/presence/negative/*.jpg
  dataset/presence/headset/*.jpg
    dataset/color/green/*.jpg, blue/*.jpg, red/*.jpg
"""

from __future__ import annotations

import argparse
import configparser
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generate_datapoints import augment_dataset
from silent_disco import COLORS, SmallConvNet, resolve_device, GenerateSettings
from trainHead import train_head_detector
from training_settings import load_training_settings

HEAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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
        return torch.from_numpy(array.transpose(2, 0, 1)), label# type: ignore


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
            optimizer.step()# type: ignore
            samples_seen += len(images)
            total_loss += loss.item() * len(images)
            progress.set_postfix(loss=f"{loss.item():.4f}")# type: ignore
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
    parser.add_argument("--kind", choices=("head", "presence", "color"), help="Train one model; omit to train all models.")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Model file for one kind, or output directory when training both.")
    parser.add_argument("--settings", type=Path, default=Path("generate.settings"), help="Augmentation settings file.")
    parser.add_argument("--training-settings", type=Path, default=Path("training.settings"), help="Shared YOLO training and benchmark settings file.")
    parser.add_argument("--processed-dataset", type=Path, default=Path("dataset_processed"), help="Temporary on-disk augmented dataset.")
    parser.add_argument("--reuse-processed-dataset", action="store_true", help="Reuse an existing generated head dataset instead of rebuilding it.")
    parser.add_argument("--keep-processed-dataset", action="store_true", help="Keep generated data after training.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm", "xpu"), default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Maximum epochs; 0 means no epoch limit.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=None, help="YOLO head-detector batch size; keep small on ROCm.")
    parser.add_argument("--head-image-size", type=int, default=None, help="YOLO head-detector training image size.")
    parser.add_argument("--target-loss", type=float, help="Stop when the epoch average loss reaches this value.")
    parser.add_argument("--max-time", type=float, help="Stop after this many seconds.")
    parser.add_argument("--isSubprocess", type=bool, default=False, help="Indicates if this is a subprocess. and prevents ROCM re-invocation. this should only be called automatically by the script itself, not by the user.")
    args = parser.parse_args()
    generate = load_generate_settings(args.settings)
    training = load_training_settings(args.training_settings)
    epochs = training.epochs if args.epochs is None else args.epochs
    device = training.device if args.device is None else args.device
    batch_size = training.batch_size if args.head_batch_size is None else args.head_batch_size
    image_size = training.image_size if args.head_image_size is None else args.head_image_size
    kinds = (args.kind,) if args.kind else ("presence", "color","head")
    if len(kinds) > 1 and args.output.suffix:
        parser.error("--output must be a directory when --kind is omitted")
    print(f"Training models for [{', '.join(kinds)}]", flush=True)
    print(f"Generating processed dataset at {args.processed_dataset}...", flush=True)
    for kind in kinds:
        if kind not in ("presence", "color"):
            continue
        source_labels = ("negative", "headset") if kind == "presence" else COLORS
        processed_kind = args.processed_dataset / kind
        if processed_kind.exists() and any(processed_kind.iterdir()):
            print(f"{kind}: reusing existing processed data at {processed_kind}", flush=True)
        else:
            originals, generated = augment_dataset(args.dataset / kind, processed_kind, generate.iterations if generate.enabled else 0, generate.noise_strength, generate.max_shift, generate.smoke_strength, generate.seed, True, source_labels)
            print(f"{kind}: copied={originals} augmented={generated}", flush=True)
    for kind in kinds:
        print(f"Training {kind} model...")
        if kind in ("head"):
            train_head_detector(args.dataset / "heads", args.processed_dataset / "head", epochs, generate, device, batch_size, image_size, training.model, training.amp, training.workers, args.isSubprocess, args.reuse_processed_dataset)
            continue
        labels = ("negative", "headset") if kind == "presence" else COLORS
        output = args.output / f"{kind}.pt" if args.output.is_dir() or not args.output.suffix else args.output
        train(args.processed_dataset / kind, labels, output, epochs, args.batch_size, args.target_loss, args.max_time, device)
        print(f"Finished training {kind} model.")
    if generate.delete_dataset_after_training and not args.keep_processed_dataset and args.processed_dataset.is_dir():
        print("Deleting temporary dataset after training...")
        shutil.rmtree(args.processed_dataset, ignore_errors=True)
        args.processed_dataset.mkdir(parents=True, exist_ok=True)
        (args.processed_dataset / ".gitkeep").touch()
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