"""Train the two small models used by silent_disco.py.

Expected dataset layout:
  dataset/presence/negative/*.jpg
  dataset/presence/headset/*.jpg
  dataset/color/green/*.jpg, blue/*.jpg, red/*.jpg, unknown/*.jpg
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from silent_disco import COLOR_CLASSES, SmallConvNet, resolve_device


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


def wait_for_enter(stop_event: threading.Event) -> None:
    try:
        input()
        stop_event.set()
    except EOFError:
        return


def train(root: Path, labels: tuple[str, ...], output: Path, epochs: int, batch_size: int, target_loss: float | None, max_time: float | None) -> None:
    dataset = CropDataset(root, labels)
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
        for images, targets in loader:
            if stop_event.is_set() or (max_time is not None and time.monotonic() - started >= max_time):
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        average_loss = total_loss / len(dataset)
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
    parser.add_argument("--kind", choices=("presence", "color"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs; 0 means no epoch limit.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-loss", type=float, help="Stop when the epoch average loss reaches this value.")
    parser.add_argument("--max-time", type=float, help="Stop after this many seconds.")
    args = parser.parse_args()
    labels = ("negative", "headset") if args.kind == "presence" else COLOR_CLASSES
    train(args.dataset / args.kind, labels, args.output, args.epochs, args.batch_size, args.target_loss, args.max_time)


if __name__ == "__main__":
    main()