"""Train the YOLO head detector from dataset/heads annotations."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    epochs = min(15, max(1, args.epochs))
    yaml_path = args.dataset / "data.yaml"
    yaml_path.write_text(
        f"path: {args.dataset.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: head\n"
    )
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(data=str(yaml_path), epochs=epochs, imgsz=1280, device="cpu", project="models", name="head-detector", exist_ok=True)
    print("saved head detector under models/head-detector/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
