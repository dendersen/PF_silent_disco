"""Compare SmallConvNet training-step speed across available devices."""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from benchmark_ai import requested_devices, synchronize
from silent_disco import SmallConvNet


def benchmark(device: torch.device, batch_size: int, warmup: int, iterations: int) -> tuple[float, float]:
    model = SmallConvNet(2).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 3, 255, 255, device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
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
    milliseconds_per_step = elapsed * 1000.0 / iterations
    samples_per_second = batch_size * iterations / elapsed
    return milliseconds_per_step, samples_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SmallConvNet training throughput.")
    parser.add_argument("--device", choices=("all", "cpu", "cuda", "rocm", "xpu"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch size must be positive, warmup cannot be negative, and iterations must be positive")
    print(f"batch={args.batch_size} warmup={args.warmup} iterations={args.iterations}")
    results = []
    for name, device in requested_devices(args.device):
        milliseconds, samples_per_second = benchmark(device, args.batch_size, args.warmup, args.iterations)
        results.append((samples_per_second, name, milliseconds))
        print(f"{name}: {milliseconds:.2f} ms/step, {samples_per_second:.2f} samples/s")
    fastest = max(results)
    print(f"fastest: {fastest[1]} ({fastest[0]:.2f} samples/s)")


if __name__ == "__main__":
    main()
