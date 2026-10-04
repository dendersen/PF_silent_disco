"""Benchmark YOLO training-step throughput using shared training settings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from benchmark_ai import requested_devices, synchronize
from silent_disco import SmallConvNet, resolve_device
from training_settings import TrainingSettings, load_training_settings


def _scalar_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float().mean()
    if isinstance(output, (tuple, list)):
        tensors = [_scalar_output(item) for item in output if item is not None]
        if tensors:
            return sum(tensors)
    if isinstance(output, dict):
        tensors = [_scalar_output(item) for item in output.values() if item is not None]
        if tensors:
            return sum(tensors)
    raise TypeError("YOLO model returned no tensor output")


def benchmark_yolo(settings: TrainingSettings, device: torch.device, batch_size: int, warmup: int, iterations: int) -> tuple[float, float]:
    from ultralytics import YOLO

    model = YOLO(str(settings.model)).model.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    images = torch.randn(batch_size, 3, settings.image_size, settings.image_size, device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=settings.amp):
            loss = _scalar_output(model(images))
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    milliseconds_per_step = elapsed * 1000.0 / iterations
    samples_per_second = batch_size * iterations / elapsed
    return milliseconds_per_step, samples_per_second


def benchmark_conv(settings: TrainingSettings, device: torch.device, batch_size: int, warmup: int, iterations: int) -> tuple[float, float]:
    model = SmallConvNet(2).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    images = torch.randn(batch_size, 3, settings.benchmark_conv_image_size, settings.benchmark_conv_image_size, device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device)
    loss_function = torch.nn.CrossEntropyLoss()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=settings.amp):
            loss = loss_function(model(images), targets)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / iterations, batch_size * iterations / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO training-step throughput.")
    parser.add_argument("--settings", type=Path, default=Path("training.settings"))
    parser.add_argument("--device", choices=("all", "auto", "cpu", "cuda", "rocm", "xpu"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--model", choices=("all", "yolo", "conv"), default=None)
    args = parser.parse_args()
    settings = load_training_settings(args.settings)
    device_name = args.device or settings.benchmark_device
    batch_size = args.batch_size or settings.batch_size
    warmup = settings.benchmark_warmup if args.warmup is None else args.warmup
    iterations = settings.benchmark_iterations if args.iterations is None else args.iterations
    models = settings.benchmark_models if args.model is None or args.model == "all" else (args.model,)
    if batch_size < 1 or warmup < 0 or iterations < 1:
        parser.error("batch size must be positive, warmup cannot be negative, and iterations must be positive")
    devices = requested_devices("all") if device_name == "all" else [(device_name, resolve_device(device_name))]
    print(f"models={','.join(models)} imgsz={settings.image_size} batch={batch_size} amp={settings.amp} warmup={warmup} iterations={iterations}")
    for name, device in devices:
        for model_name in models:
            benchmark_function = benchmark_yolo if model_name == "yolo" else benchmark_conv
            milliseconds, samples = benchmark_function(settings, device, batch_size, warmup, iterations)
            print(f"{model_name} {name}: {milliseconds:.2f} ms/step, {samples:.2f} samples/s")


if __name__ == "__main__":
    main()