"""Shared YOLO training and benchmark settings."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingSettings:
    model: Path = Path("models/yolo11n.pt")
    image_size: int = 640
    batch_size: int = 1
    device: str = "auto"
    amp: bool = True
    workers: int = 0
    epochs: int = 12
    benchmark_warmup: int = 5
    benchmark_iterations: int = 20
    benchmark_device: str = "all"
    benchmark_models: tuple[str, ...] = ("yolo", "conv")
    benchmark_conv_image_size: int = 255


def load_training_settings(path: Path = Path("training.settings")) -> TrainingSettings:
    settings = TrainingSettings()
    parser = configparser.ConfigParser()
    if path.exists():
        parser.read(path)
    yolo = parser["yolo"] if parser.has_section("yolo") else {}
    benchmark = parser["benchmark"] if parser.has_section("benchmark") else {}
    result = TrainingSettings(
        model=Path(yolo.get("model", str(settings.model))),
        image_size=int(yolo.get("imgsz", settings.image_size)),
        batch_size=int(yolo.get("batch", settings.batch_size)),
        device=yolo.get("device", settings.device),
        amp=yolo.getboolean("amp", fallback=settings.amp),
        workers=int(yolo.get("workers", settings.workers)),
        epochs=int(yolo.get("epochs", settings.epochs)),
        benchmark_warmup=int(benchmark.get("warmup", settings.benchmark_warmup)),
        benchmark_iterations=int(benchmark.get("iterations", settings.benchmark_iterations)),
        benchmark_device=benchmark.get("device", settings.benchmark_device),
        benchmark_models=tuple(item.strip() for item in benchmark.get("models", ",".join(settings.benchmark_models)).split(",") if item.strip()),
        benchmark_conv_image_size=int(benchmark.get("conv_imgsz", settings.benchmark_conv_image_size)),
    )
    if result.image_size < 1 or result.batch_size < 1 or result.workers < 0 or result.epochs < 1:
        raise ValueError("training.settings requires positive imgsz, batch, and epochs; workers cannot be negative")
    if result.benchmark_warmup < 0 or result.benchmark_iterations < 1 or result.benchmark_conv_image_size < 1:
        raise ValueError("training.settings requires non-negative benchmark warmup, positive iterations, and positive conv_imgsz")
    if not result.benchmark_models or any(model not in {"yolo", "conv"} for model in result.benchmark_models):
        raise ValueError("training.settings benchmark models must contain only yolo and conv")
    return result