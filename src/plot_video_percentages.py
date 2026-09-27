"""Run headset inference on videos and plot recognized color percentages over time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

if matplotlib.get_backend().lower().endswith("agg"):
    try:
        matplotlib.use("QtAgg")
    except ImportError as error:
        raise RuntimeError("Live plotting requires PySide6; install dependencies from src/requirements.txt") from error

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import torch
from tqdm import tqdm

from silent_disco import COLORS, analyze_frame, load_model, load_person_detector, resolve_device


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


class LivePlot:
    def __init__(self, videos: list[Path], update_every: int) -> None:
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(12, 7))
        self.figure.subplots_adjust(right=0.78)
        self.axis.set_xlabel("Time (seconds)")
        self.axis.set_ylabel("Recognized headset percentage")
        self.axis.set_ylim(0, 100)
        self.axis.grid(True, alpha=0.25)
        self.lines = {
            (video.name, color): self.axis.plot([], [], label=f"{video.name}: {color}")[0]
            for video in videos
            for color in COLORS
        }
        self.video_names = [video.name for video in videos]
        self.active_video = self.video_names[0]
        for (video_name, _), line in self.lines.items():
            line.set_visible(video_name == self.active_video)
        self.pending = 0
        self.update_every = max(1, update_every)
        self.axis.legend()
        selector_axis = self.figure.add_axes((0.8, 0.2, 0.18, 0.6))
        self.selector = RadioButtons(selector_axis, self.video_names, active=0)
        self.selector.on_clicked(self.select_video)
        self.figure.show()
        plt.pause(0.1)

    def select_video(self, video_name: str) -> None:
        self.active_video = video_name
        for (name, _), line in self.lines.items():
            line.set_visible(name == video_name)
        self.axis.relim()
        self.axis.autoscale_view(scalex=True, scaley=False)
        self.axis.set_title(video_name)
        self.figure.canvas.draw_idle()
        plt.pause(0.001)

    def set_active_video(self, video_name: str) -> None:
        if video_name != self.active_video:
            self.selector.set_active(self.video_names.index(video_name))

    def update(self, row: dict[str, float | str], force: bool = False) -> None:
        self.pending += 1
        if not force and self.pending < self.update_every:
            return
        video = str(row["video"])
        time_seconds = float(row["time_seconds"])
        for color in COLORS:
            line = self.lines[(video, color)]
            line.set_data([*line.get_xdata(), time_seconds], [*line.get_ydata(), float(row[color])])
        self.axis.relim()
        self.axis.autoscale_view(scalex=True, scaley=False)
        self.pending = 0
        self.figure.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()

    def save(self, path: Path) -> None:
        self.figure.savefig(path, dpi=160)


def process_video(path: Path, detector, presence_model, color_model, device: torch.device, every: int, person_confidence: float, live_plot: LivePlot | None, display_ai_frame: bool) -> list[dict[str, float | str]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = (frame_count + every - 1) // every if frame_count else None
    rows: list[dict[str, float | str]] = []
    frame_number = 0
    if live_plot is not None:
        live_plot.set_active_video(path.name)
    with tqdm(total=sample_count, desc=path.name, unit="sample") as progress:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % every == 0:
                detections, ratios, annotated = analyze_frame(frame, detector, presence_model, color_model, device, person_confidence=person_confidence)
                if display_ai_frame:
                    cv2.imshow("AI frame analysis", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt
                row: dict[str, float | str] = {"video": path.name, "time_seconds": frame_number / fps}
                for color in COLORS:
                    row[color] = ratios[color] * 100.0
                row["recognized_count"] = sum(detection.color in COLORS for detection in detections)
                rows.append(row)
                if live_plot is not None:
                    live_plot.update(row)
                progress.update(1)
            frame_number += 1
    capture.release()
    if live_plot is not None and rows:
        live_plot.update(rows[-1], force=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recognized headset color percentages over video time.")
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--presence-model", type=Path, default=Path("models/presence.pt"))
    parser.add_argument("--color-model", type=Path, default=Path("models/color.pt"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--detector-device", choices=("cpu", "cuda", "rocm"), default="cpu", help="Device for YOLO person detection; CPU avoids ROCm detector crashes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm", "xpu"), default="auto")
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--plot-every", type=int, default=5, help="Refresh the live plot after this many sampled frames.")
    parser.add_argument("--display-ai-frame", action="store_true", help="Show the previous processed frame with headset classification boxes.")
    parser.add_argument("--display-width", type=int, default=1280, help="Preview window width in pixels.")
    parser.add_argument("--display-height", type=int, default=720, help="Preview window height in pixels.")
    parser.add_argument("--person-confidence", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("models/video_color_percentages.png"))
    parser.add_argument("--csv", type=Path, default=Path("models/video_color_percentages.csv"))
    args = parser.parse_args()
    videos = sorted(path for path in args.video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        parser.error(f"no videos found in {args.video_dir}")
    print(f"found {len(videos)} video(s)", flush=True)
    live_plot = LivePlot(videos, args.plot_every)
    device = resolve_device(args.device)
    print(f"loading YOLO detector on {args.detector_device}...", flush=True)
    detector_device = resolve_device(args.detector_device)
    detector = load_person_detector(args.detector_model, detector_device)
    print(f"loading presence model on {device}...", flush=True)
    presence_model = load_model(args.presence_model, 2, device)
    print(f"loading color model on {device}...", flush=True)
    color_model = load_model(args.color_model, 3, device)
    print("starting video inference...", flush=True)
    if args.display_ai_frame:
        cv2.namedWindow("AI frame analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI frame analysis", max(320, args.display_width), max(240, args.display_height))
    rows = [row for video in videos for row in process_video(video, detector, presence_model, color_model, device, max(1, args.every), args.person_confidence, live_plot, args.display_ai_frame)]
    print("writing CSV and plot...", flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "time_seconds", *COLORS, "recognized_count"))
        writer.writeheader()
        writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    live_plot.save(args.output)
    live_plot.close()
    if args.display_ai_frame:
        cv2.destroyWindow("AI frame analysis")
    print(f"processed_videos={len(videos)} samples={len(rows)} csv={args.csv} plot={args.output} device={device}")


if __name__ == "__main__":
    main()