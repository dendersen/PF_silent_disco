"""Compare SmallConvNet inference speed across available PyTorch devices."""

from __future__ import annotations

import argparse
import time

import torch

from silent_disco import SmallConvNet, rocm_available


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize()


def available_devices() -> list[tuple[str, torch.device]]:
    devices = [("CPU", torch.device("cpu"))]
    if torch.cuda.is_available():
        devices.append(("AMD ROCm" if rocm_available() else "NVIDIA CUDA", torch.device("cuda")))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append(("Intel XPU", torch.device("xpu")))
    return devices


def requested_devices(requested: str) -> list[tuple[str, torch.device]]:
    if requested == "all":
        return available_devices()
    if requested == "cpu":
        return [("CPU", torch.device("cpu"))]
    if requested in {"cuda", "rocm"}:
        if not torch.cuda.is_available() or (requested == "rocm" and not rocm_available()):
            raise RuntimeError(f"Requested benchmark device '{requested}' is not available")
        name = "AMD ROCm" if requested == "rocm" else "NVIDIA CUDA"
        return [(name, torch.device("cuda"))]
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Requested benchmark device 'xpu' is not available")
        return [("Intel XPU", torch.device("xpu"))]
    raise ValueError(f"Unknown device selection: {requested}")


def benchmark(device: torch.device, batch_size: int, warmup: int, iterations: int) -> tuple[float, float]:
    model = SmallConvNet(2).to(device).eval()
    batch = torch.randn(batch_size, 3, 255, 255, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(batch)
        synchronize(device)
    elapsed = time.perf_counter() - started
    milliseconds_per_batch = elapsed * 1000.0 / iterations
    images_per_second = batch_size * iterations / elapsed
    return milliseconds_per_batch, images_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark headset-model inference on available devices.")
    parser.add_argument("--device", choices=("all", "cpu", "cuda", "rocm", "xpu"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch size must be positive, warmup cannot be negative, and iterations must be positive")
    print(f"torch={torch.__version__} batch={args.batch_size} warmup={args.warmup} iterations={args.iterations}")
    results = []
    for name, device in requested_devices(args.device):
        started = time.perf_counter()
        milliseconds, images_per_second = benchmark(device, args.batch_size, args.warmup, args.iterations)
        results.append((images_per_second, name, milliseconds, time.perf_counter() - started))
        print(f"{name}: {milliseconds:.2f} ms/batch, {images_per_second:.2f} images/s")
    fastest = max(results)
    print(f"fastest: {fastest[1]} ({fastest[0]:.2f} images/s)")


if __name__ == "__main__":
    main()