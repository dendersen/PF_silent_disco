# Silent disco estimator

Install the common Python dependencies from this directory:

```bash
python3 -m pip install -r src/requirements.txt
```

Install PyTorch separately using the official selector for the target device:

- NVIDIA: install the CUDA wheel for the installed CUDA version.
- AMD: install `src/requirements-rocm.txt`, changing `rocm6.3` to the installed ROCm version. ROCm uses PyTorch's `cuda` device API internally, but the program detects the HIP build explicitly.
- Intel Arc: install `src/requirements-xpu.txt`. The runtime uses PyTorch's `xpu` device API.

The command defaults to `--device auto`, preferring CUDA/ROCm, then Intel XPU,
then CPU. Use `--device cuda`, `--device rocm`, `--device xpu`, or `--device cpu` to require a
specific backend; unavailable explicit devices produce an error instead of
silently running on the CPU.

On an Intel Arc Linux host, install the explicit XPU build and verify the backend before running:

```bash
python3 -m pip install -r src/requirements-xpu.txt
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
python3 src/silent_disco.py images/_MG_7098.JPG --device xpu
```

The runtime stages are in `silent_disco.py`:

1. `preprocess_image` upscales undersized frames until their shortest dimension is 3500 pixels, never downscales, and keeps colorful or bright pixels.
2. `find_candidate_points` returns intentionally generous blob proposals.
3. A two-class `SmallConvNet` decides whether each 255x255 crop contains a headset.
4. A four-class `SmallConvNet` predicts green, blue, red, or unknown.

Train checkpoints with labeled crops:

```text
dataset/
  presence/negative/*.jpg
  presence/headset/*.jpg
  color/green/*.jpg
  color/blue/*.jpg
  color/red/*.jpg
  color/unknown/*.jpg
```

Create those crops with the browser annotator. Click a headset or other point
in the image, then choose a button. `No headset` creates a negative presence
crop and an unknown color crop. Multiple points can be labeled before moving to
the next image:

The saved-crop gallery below the image lets you inspect every crop, change its
label, or remove it from both training datasets.

Use `Review auto points` to run the detector's candidate-point stage on the
current image. The annotator presents each new candidate one at a time; choose
its label and it advances automatically. After the final point, normal manual
clicking resumes for the same image.

Auto-review points are separated by at least 60 pixels in the original image,
and corner candidates are added when color blobs are sparse.

```bash
python3 src/annotate_training.py images --dataset dataset --open
```
The next-image action alternates between a randomly selected still image and a
random frame from a randomly selected video. If port `8765` is already in use,
choose another port, for example `--port 8766`.

```bash
python3 src/train_models.py dataset --kind presence --output models/presence.pt
python3 src/train_models.py dataset --kind color --output models/color.pt
```

Training stops at the first of four conditions: press Enter, reach
`--target-loss`, reach `--max-time` seconds, or reach `--epochs`. Use
`--epochs 0` for no epoch limit:

```bash
python3 src/train_models.py dataset --kind color --output models/color.pt \
  --epochs 0 --target-loss 0.20 --max-time 3600
```

The checkpoint is saved when training stops, and the terminal reports which
condition ended the run. Press Enter in the training terminal to stop manually.

Compare AI inference speed across the CPU and available accelerators. The
benchmark uses the same `SmallConvNet` architecture and 255x255 input size as
the detector:

```bash
python3 src/benchmark_ai.py
```

Use `--device cpu`, `--device xpu`, `--device cuda`, or `--device rocm` to test
one backend. Increase `--iterations` for a more stable result.

Benchmark training performance, including forward, backward, and optimizer
steps:

```bash
python3 src/benchmark_training.py
```

This uses synthetic data and reports training samples per second. It does not
measure image loading or preprocessing.

Run an image or sample every third video frame. Each output line is JSON, which
makes it easy to feed the ratios into a dashboard or event logger:

```bash
python3 src/silent_disco.py images/_MG_7098.JPG \
  --presence-model models/presence.pt \
  --color-model models/color.pt

python3 src/silent_disco.py images/GX010811.MP4 --every 3 --display
```

`--display` is optional. The reported `smoothed_ratios` use an exponential moving
average to reduce frame-to-frame jitter in semi-real-time video.