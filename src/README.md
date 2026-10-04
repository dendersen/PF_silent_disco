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

The color and presence classifiers use a widened 32→64→128 channel CNN. Existing checkpoints from the previous smaller architecture must be retrained before inference.

On an Intel Arc Linux host, install the explicit XPU build and verify the backend before running:

```bash
python3 -m pip install -r src/requirements-xpu.txt
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
python3 src/silent_disco.py images/_MG_7098.JPG --device xpu
```

The runtime stages are in `silent_disco.py`:

1. `preprocess_image` upscales undersized frames until their shortest dimension is 3500 pixels, never downscales, and keeps colorful or bright pixels.
2. Ultralytics YOLO detects people and returns bounding boxes, avoiding color/blob proposals that can miss people or select lights and walls.
3. Each person box is expanded to a 1:1 square and resized to 255x255 for the next stages.
4. The four-class `SmallConvNet` predicts green, blue, red, or unknown. Detector boxes are retained even when color is unknown so crops are not silently discarded.

The default detector is `models/yolo11n.pt`; Ultralytics downloads it on first use. The default person confidence is `0.08`, which is useful for overhead event photos. Adjust it with `--person-confidence`. The older presence checkpoint is not used as a gate by default because it was trained on point-centered crops; enable `--verify-presence` only after retraining it with detector-generated square crops.

Train checkpoints with labeled crops:

```text
dataset/
  presence/negative/*.jpg
  presence/headset/*.jpg
  color/green/*.jpg
  color/blue/*.jpg
  color/red/*.jpg
```

Before training, augmented crops are written to `dataset_processed` from `generate.settings`. Each variant can add bounded pixel noise, move the crop center in both axes, and simulate smoke by reducing saturation and increasing brightness:

```ini
[generate]
enabled = true
iterations = 4
noise_strength = 0.15
max_shift = 10
smoke_strength = 0.15
seed = 20260926
delete_dataset_after_training = true
```

Train both models with one command, or select one with `--kind`:

```bash
python3 src/train_models.py dataset
python3 src/train_models.py dataset --kind color --output models/color.pt
```

For ROCm training, use the ROCm PyTorch environment explicitly:

```bash
python3 src/train_models.py dataset --device rocm --epochs 25
```

YOLO training and the YOLO training-step benchmark share `training.settings`:

```ini
[yolo]
model = models/yolo11n.pt
imgsz = 640
batch = 1
device = auto
amp = false
workers = 0
epochs = 12

[benchmark]
device = all
models = yolo, conv
conv_imgsz = 255
warmup = 5
iterations = 20
```

Run the benchmark with the same settings used by training:

```bash
python3 src/benchmark_training.py
```

This benchmarks both YOLO and `SmallConvNet` on every selected device. Use
`--model yolo` or `--model conv` to benchmark only one model family.

Override the file for a one-off comparison, for example `--device xpu` or
`--batch-size 2`. Training accepts `--training-settings PATH` when using a
different configuration file.

After training, generate the color-percentage CSV and plot for every video in a directory:

```bash
python3 src/plot_video_percentages.py images --device rocm --detector-device cpu --every 3
```

Add `--display-ai-frame` to open a live annotated frame window. Each detected human has a yellow outer box. The smaller, slightly filled inner box is the exact crop sent to the AI: dark gray when rejected by the presence model, or blue, green, or red when accepted and classified as that DJ color. The preview defaults to 1280×720; adjust it with `--display-width` and `--display-height`. Inference still uses the full-resolution frame. Press Escape to stop frame processing.

The custom presence and color models run on ROCm. YOLO person detection defaults to CPU because its ROCm path can segfault on some PyTorch/Ultralytics combinations; use `--detector-device rocm` only when that stack is known to be stable. Unknown/no-color detections are excluded from the percentage denominator. The live window has a video selector on the right; choosing a video shows only that video's curves while processing continues.

Each epoch displays a batch progress bar. The original `dataset` is never modified. The processed dataset remains on disk while all requested models train and is removed afterward when `delete_dataset_after_training` is enabled. Use `--processed-dataset` to choose another generated-data location.

Existing processed presence and color data is reused automatically. To reuse an existing generated head dataset as well, keep it and pass:

```bash
python3 src/train_models.py dataset --kind head \
  --reuse-processed-dataset --keep-processed-dataset
```

Create training data with the browser dataset studio. It has a head trainer for
YOLO boxes and a color/presence trainer for one detector candidate at a time.
Both trainers choose random video frames about 80% of the time, while retaining
still images for coverage. `Skip` records no training example. The home page
shows counts for every label, cumulative classifier accuracy, and CPU-only
training controls capped at 15 epochs:

The saved-crop gallery below the image lets you inspect every crop, change its
label, or remove it from both training datasets.

The head trainer writes YOLO data under `dataset/heads`. If port `8765` is
already in use, choose another port with `--port 8766`. After training a head
detector, restart the studio with its checkpoint using
`--head-detector-model models/head-detector/weights/best.pt`.

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