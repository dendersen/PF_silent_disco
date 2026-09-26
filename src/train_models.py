"""Train the two small models used by silent_disco.py.

Expected dataset layout:
  dataset/presence/negative/*.jpg
  dataset/presence/headset/*.jpg
  dataset/color/green/*.jpg, blue/*.jpg, red/*.jpg, unknown/*.jpg
"""

from __future__ import annotations

import argparse
import configparser
import shutil
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

from generate_datapoints import augment_image
from silent_disco import COLOR_CLASSES, SmallConvNet, resolve_device


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
    def __init__(self, root: Path, labels: tuple[str, ...], generate: GenerateSettings) -> None:
        self.items = [(path, index) for index, label in enumerate(labels) for path in sorted((root / label).glob("*"))]
        self.generate = generate
        self.base_length = len(self.items)

    def __len__(self) -> int:
        multiplier = self.generate.iterations + 1 if self.generate.enabled else 1
        return self.base_length * multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source_index = index % self.base_length
        iteration = index // self.base_length
        path, label = self.items[source_index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read {path}")
        if self.generate.enabled and iteration:
            image = augment_image(
                image,
                np.random.default_rng(self.generate.seed + index),
                self.generate.noise_strength,
                self.generate.max_shift,
                self.generate.smoke_strength,
            )
        image = cv2.cvtColor(cv2.resize(image, (255, 255)), cv2.COLOR_BGR2RGB)
        array = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(array.transpose(2, 0, 1)), label


def wait_for_enter(stop_event: threading.Event) -> None:
    try:
        input()
        stop_event.set()
    except EOFError:
        return


def train(root: Path, labels: tuple[str, ...], output: Path, epochs: int, batch_size: int, target_loss: float | None, max_time: float | None, generate: GenerateSettings) -> None:
    dataset = CropDataset(root, labels, generate)
    if not dataset:
        raise ValueError(f"No training crops found under {root}")
    device = resolve_device("auto")
    model = SmallConvNet(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    stop_event = threading.Event()
    threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True).start()
    started = time.monotonic()
    epoch = 0
    stop_reason = "user"
    print(f"Training {len(dataset)} crops ({len(dataset) // (generate.iterations + 1 if generate.enabled else 1)} originals, augmentation={'on' if generate.enabled else 'off'})")
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
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs or '∞'}", unit="batch")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--kind", choices=("presence", "color"), help="Train one model; omit to train both.")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Model file for one kind, or output directory when training both.")
    parser.add_argument("--settings", type=Path, default=Path("generate.settings"), help="Augmentation settings file.")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs; 0 means no epoch limit.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-loss", type=float, help="Stop when the epoch average loss reaches this value.")
    parser.add_argument("--max-time", type=float, help="Stop after this many seconds.")
    args = parser.parse_args()
    generate = load_generate_settings(args.settings)
    kinds = (args.kind,) if args.kind else ("presence", "color")
    if len(kinds) > 1 and args.output.suffix:
        parser.error("--output must be a directory when --kind is omitted")
    for kind in kinds:
        labels = ("negative", "headset") if kind == "presence" else COLOR_CLASSES
        output = args.output if args.kind else args.output / f"{kind}.pt"
        train(args.dataset / kind, labels, output, args.epochs, args.batch_size, args.target_loss, args.max_time, generate)
    if generate.delete_dataset_after_training and args.dataset.name == "dataset_processed" and args.dataset.is_dir():
        shutil.rmtree(args.dataset)
        print(f"deleted temporary dataset {args.dataset}")


if __name__ == "__main__":
    main()