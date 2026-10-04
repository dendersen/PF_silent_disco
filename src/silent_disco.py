"""Semi-real-time silent disco headset audience estimator."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, cast
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.engine.results import Results

import cv2
import numpy as np
import torch
from torch import nn


COLORS = ("green", "blue", "red")
COLOR_CLASSES = ("green", "blue", "red", "unknown")
COLOR_MODEL_CLASSES = COLORS

@dataclass(frozen=True)
class GenerateSettings:
    enabled: bool = True
    iterations: int = 4
    noise_strength: float = 0.15
    max_shift: int = 10
    smoke_strength: float = 0.15
    seed: int = 20260926
    delete_dataset_after_training: bool = True

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
        return torch.from_numpy(array.transpose(2, 0, 1)), label# type: ignore

@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    box: tuple[int, int, int, int]
    presence: float
    color: str
    color_confidence: float


@dataclass
class RatioSmoother:
    alpha: float = 0.35
    values: dict[str, float] | None = None

    def update(self, counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        current = {color: (counts.get(color, 0) / total if total else 0.0) for color in COLORS}
        if self.values is None:
            self.values = current
        else:
            self.values = {color: self.alpha * current[color] + (1.0 - self.alpha) * self.values[color] for color in COLORS}
        return dict(self.values)


def rocm_available() -> bool:
    """Return true only for a PyTorch build compiled with HIP/ROCm."""
    return torch.version.hip is not None and torch.cuda.is_available()


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CPU, NVIDIA CUDA, AMD ROCm, or Intel XPU at runtime.
    requested: "auto" (default), "cpu", "cuda", "rocm", or "xpu"
    """
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("rocm", "auto") and rocm_available():
        print(f"Using ROCm device", torch.cuda.get_device_name(torch.cuda.current_device()))
        return torch.device("cuda")
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        if rocm_available():
            if requested == "cuda":
                print(f"preventing cuda use as this is a rocm build")
                print(f"there are important fixes for rocm in the codebase")
                print(f"this is a known issue with ROCm builds of PyTorch")
                print(f"please use --device rocm to run on a ROCm device")
                print(f"if you are trying to run on a CUDA device, please use different build of PyTorch")
        else:
            return torch.device("cuda")
    if requested in {"xpu", "auto"} and hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Using Intel XPU device", torch.xpu.get_device_name(torch.xpu.current_device()))
        return torch.device("xpu")
    if requested != "auto":
        raise RuntimeError(f"Requested device '{requested}' is not available in this PyTorch installation")
    print("Using CPU device")
    return torch.device("cpu")


def preprocess_image(image_bgr: np.ndarray, min_side: int = 3500) -> tuple[np.ndarray, np.ndarray]:
    """Suppress low-information pixels without downscaling the input."""
    if image_bgr is None or image_bgr.ndim != 3:# type: ignore
        raise ValueError("image_bgr must be a color image")
    height, width = image_bgr.shape[:2]
    scale = max(1.0, min_side / min(height, width))
    image = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC) if scale > 1 else image_bgr.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    colorful = ((saturation >= 70) & (value >= 45)).astype(np.uint8) * 255
    bright_neutral = ((value >= 175) & (saturation >= 20)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(colorful, bright_neutral)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return cv2.bitwise_and(enhanced, enhanced, mask=mask), mask


def find_candidate_points(cleaned_bgr: np.ndarray, mask: np.ndarray, max_candidates: int = 250, min_distance: int | None = None, augment_corners: bool = False) -> list[Candidate]:
    """Return generous point proposals; the presence model removes light artifacts."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    proposals: list[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if area < 8 or box_width < 3 or box_height < 3 or area > width * height * 0.08:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x, center_y = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        roi = cleaned_bgr[y:y + box_height, x:x + box_width]
        brightness = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))) if roi.size else 0.0
        score = min(1.0, 0.45 * min(1.0, area / 500.0) + 0.55 * brightness / 255.0)
        proposals.append(Candidate(center_x, center_y, score))
    proposals.sort(key=lambda item: item.score, reverse=True)
    selected: list[Candidate] = []
    min_distance = min_distance or max(5, round(min(height, width) * 0.006))
    for proposal in proposals:
        if all((proposal.x - item.x) ** 2 + (proposal.y - item.y) ** 2 >= min_distance**2 for item in selected):
            selected.append(proposal)
        if len(selected) >= max_candidates:
            break
    if not selected or augment_corners:
        gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_candidates, qualityLevel=0.005, minDistance=min_distance, mask=mask)
        if corners is not None:# type: ignore
            for corner in corners.reshape(-1, 2):
                point_x, point_y = round(float(corner[0])), round(float(corner[1]))
                if all((point_x - item.x) ** 2 + (point_y - item.y) ** 2 >= min_distance**2 for item in selected):
                    selected.append(Candidate(point_x, point_y, float(gray[point_y, point_x]) / 255.0))
                if len(selected) >= max_candidates:
                    break
    return selected


def crop_255(image_bgr: np.ndarray, x: int, y: int, size: int = 255) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(image_bgr, half, half, half, half, cv2.BORDER_REFLECT_101)
    return padded[y:y + size, x:x + size]

def crop_person_head(image_bgr: np.ndarray, box: tuple[int, int, int, int], size: int = 255) -> tuple[np.ndarray, tuple[int, int]]:
    """Expand a YOLO person box to a square and resize it for the classifiers."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    crop_size = max(1, x2 - x1, y2 - y1)
    half = crop_size // 2
    crop = cv2.copyMakeBorder(image_bgr, half, half, half, half, cv2.BORDER_REFLECT_101)
    crop = crop[center_y:center_y + size, center_x:center_x + size]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), (center_x, center_y)


def person_head_crop_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return a square expanded from the YOLO person box."""
    x1, y1, x2, y2 = box
    crop_size = max(1, x2 - x1, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half = crop_size // 2
    square_x1 = round(center_x - half)
    square_y1 = round(center_y - half)
    return square_x1, square_y1, square_x1 + crop_size, square_y1 + crop_size


class SmallConvNet(nn.Module):
    """Compact crop classifier with enough capacity for detector-centered crops."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(batch))


def load_model(path: Path | None, classes: int, device: torch.device) -> SmallConvNet | None:
    if path is None or not path.exists():
        return None
    model = SmallConvNet(classes).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint {path} was built with an older model size; retrain it with src/train_models.py") from error
    model.eval()
    return model


def load_person_detector(path: Path, device: torch.device | None = None):
    """Load a pretrained detector and configure it to return people only."""
    detector = YOLO(str(path))
    if device is not None:
        detector.to(str(device))
    return detector


def detect_people(
    detector:YOLO,
    image_bgr: np.ndarray,
    confidence: float = 0.08,
    image_size: int = 1280,
    tile_size: int = 3000,
    tile_overlap: float = 0.2,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return person boxes using overlapping tiles so small people are not lost."""
    height, width = image_bgr.shape[:2]
    tile_size = max(0, tile_size)
    tiles: list[tuple[int, int, np.ndarray]] = []
    if not tile_size or (width <= tile_size and height <= tile_size):
        tiles = [(0, 0, image_bgr)]
    else:
        step = max(1, round(tile_size * (1.0 - min(0.8, max(0.0, tile_overlap)))))
        tiles = []
        bottom = height
        for top in range(0, height, step):
            for left in range(0, width, step):
                right, bottom = min(width, left + tile_size), min(height, top + tile_size)
                tiles.append((left, top, image_bgr[top:bottom, left:right]))
                if right == width:
                    break
            if bottom == height:
                break

    boxes: list[list[int]] = []
    scores: list[float] = []
    for left, top, tile in tiles:
        result:Results|torch.Tensor = detector.predict(tile, classes=[0], conf=confidence, imgsz=image_size, max_det=300, verbose=False)[0] # type: ignore
        if not isinstance(result, torch.Tensor) and not isinstance(result, Results):
            raise RuntimeError("Unexpected YOLO result type; please update ultralytics to a recent version")
        for coordinates, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):# type: ignore
            x1, y1, x2, y2 = (round(float(value)) for value in coordinates)
            box:list[int] = [max(0, x1 + left), max(0, y1 + top), min(width - 1, x2 + left), min(height - 1, y2 + top)]
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
                scores.append(float(score))
    selected:Sequence[int]  = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45) if boxes else []
    return [((boxes[index][0], boxes[index][1], boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3]), scores[index]) for index in (int(item) for item in selected)]


def tensor_from_bgr(crop: np.ndarray, device: torch.device) -> torch.Tensor:
    return tensors_from_bgr([crop], device)


def tensors_from_bgr(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    if not crops:
        return torch.empty((0, 3, 255, 255), device=device)
    arrays = [((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1) for crop in crops]
    return torch.from_numpy(np.stack(arrays)).to(device)# type: ignore


def predict_probabilities(model: SmallConvNet, crops: list[np.ndarray], device: torch.device, batch_size: int = 32) -> torch.Tensor:
    outputs:list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = tensors_from_bgr(crops[start:start + batch_size], device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu())
    if outputs:
        return torch.cat(outputs) 
    else:
        output_layer = cast(nn.Linear, model.classifier[-1])
        return torch.empty((0, output_layer.out_features))

def analyze_frame(image_bgr: np.ndarray, detector:YOLO, presence_model: SmallConvNet | None, color_model: SmallConvNet | None, device: torch.device, presence_threshold: float = 0.5, person_confidence: float = 0.08, verify_presence: bool = True, detector_image_size: int = 1280, detector_tile_size: int = 3000, detector_tile_overlap: float = 0.2) -> tuple[list[Detection], dict[str, float], np.ndarray]:
    people = detect_people(detector, image_bgr, person_confidence, detector_image_size, detector_tile_size, detector_tile_overlap)
    crops:list[np.ndarray] = []
    points:list[tuple[int, int]] = []
    boxes:list[tuple[int, int, int, int]] = []
    presence_scores:list[float] = []
    detector_scores:list[float] = []
    for box, detector_score in people:
        crop, point = crop_person_head(image_bgr, box)
        boxes.append(box)
        points.append(point)
        crops.append(crop)
        detector_scores.append(detector_score)
    if presence_model is None or not verify_presence:
        presence_scores = detector_scores
    else:
        presence_scores = predict_probabilities(presence_model, crops, device)[:, 1].tolist() # type: ignore 
    accepted = [(point, box, crop, presence) for point, box, crop, presence in zip(points, boxes, crops, presence_scores) if not verify_presence or presence >= presence_threshold]
    if color_model is None:
        raise RuntimeError("Color model is required for color prediction")
    elif accepted and presence_model is not None and verify_presence:
        probabilities = predict_probabilities(color_model, [crop for _, _, crop, _ in accepted], device)
        color_predictions = [(COLOR_MODEL_CLASSES[int(torch.argmax(probability).item())], float(torch.max(probability).item())) for probability in probabilities]
    else:
        color_predictions = [("unknown", 0.0) for _ in accepted]
    detections: list[Detection] = []
    for (point, box, _, presence), (color, color_confidence) in zip(accepted, color_predictions):
        x, y = point
        detections.append(Detection(x, y, box, presence, color, color_confidence))
    counts = {color: sum(detection.color == color for detection in detections) for color in COLORS}
    ratios = {color: counts[color] / max(1, sum(counts.values())) for color in COLORS}
    annotated = image_bgr.copy()
    colors_bgr = {"green": (0, 200, 0), "blue": (220, 80, 0), "red": (0, 0, 220)}
    accepted_by_box = {detection.box: detection for detection in detections}
    for box, _ in people:
        detection = accepted_by_box.get(box)
        x1, y1, x2, y2 = box
        color = colors_bgr.get(detection.color, (70, 70, 70)) if detection else (70, 70, 70)
        label = detection.color if detection else "no headset"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        head_x1, head_y1, head_x2, head_y2 = person_head_crop_box(box)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return detections, ratios, annotated


def iter_frames(path: Path, every: int) -> Iterable[tuple[int, np.ndarray]]:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read {path}")
        yield 0, image
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % every == 0:
            yield frame_number, frame
        frame_number += 1
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate silent disco DJ audience ratios from images or video.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--presence-model", type=Path)
    parser.add_argument("--color-model", type=Path)
    parser.add_argument("--detector-model", type=Path, default=Path("models/yolo11n.pt"), help="YOLO person detector checkpoint; downloaded by Ultralytics when absent.")
    parser.add_argument("--person-confidence", type=float, default=0.08)
    parser.add_argument("--skip-presence", action="store_true", help="Skip presence-model filtering; color prediction will receive every detector crop.")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--every", type=int, default=3, help="Analyze every Nth video frame.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm", "xpu"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    detector = load_person_detector(args.detector_model, device)
    presence_model = load_model(args.presence_model, 2, device)
    color_model = load_model(args.color_model, len(COLOR_MODEL_CLASSES), device)
    smoother = RatioSmoother()
    for frame_number, frame in iter_frames(args.input, max(1, args.every)):
        started = time.perf_counter()
        detections, ratios, annotated = analyze_frame(frame, detector, presence_model, color_model, device, presence_threshold=args.presence_threshold, person_confidence=args.person_confidence, verify_presence=not args.skip_presence)
        counts = {color: sum(d.color == color for d in detections) for color in COLORS}
        payload:dict[str, int|float|dict[str, int|float]] = {"frame": frame_number, "detections": len(detections), "ratios": ratios, "smoothed_ratios": smoother.update(counts), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        print(json.dumps(payload), flush=True)
        if args.display:
            cv2.imshow("silent disco", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()