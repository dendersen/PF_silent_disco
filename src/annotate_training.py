"""Browser annotation tool for creating headset training crops."""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch

from silent_disco import COLOR_MODEL_CLASSES, crop_255, crop_person_head, detect_people, load_model, load_person_detector, predict_probabilities, resolve_device


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silent disco training points</title>
<style>
body { font-family: sans-serif; margin: 24px auto; max-width: 1100px; color: #17202a; }
h1 { margin-bottom: 4px; } #status { color: #52606d; margin-bottom: 16px; }
#image-wrap { position: relative; width: fit-content; max-width: 100%; }
#frame { max-width: 100%; border: 1px solid #ccd6dd; cursor: crosshair; display: block; }
.selection { display: none; position: absolute; pointer-events: none; border: 3px solid #f4d03f; box-sizing: border-box; }
.selection.negative { background: rgba(86, 101, 115, .32); border-color: #566573; }
.selection.green { background: rgba(21, 153, 87, .32); border-color: #159957; }
.selection.blue { background: rgba(36, 99, 235, .32); border-color: #2463eb; }
.selection.red { background: rgba(211, 63, 73, .32); border-color: #d33f49; }
#zoom-panel { display: none; margin-top: 16px; } #zoom { width: 510px; max-width: 100%; image-rendering: auto; border: 2px solid #ccd6dd; } #prediction-note { color: #52606d; font-weight: bold; margin-top: 8px; }
#buttons { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
button { border: 0; border-radius: 4px; color: white; cursor: pointer; font-size: 16px; padding: 12px 18px; }
button:disabled { cursor: not-allowed; opacity: .45; } .negative { background: #566573; }
.green { background: #159957; } .blue { background: #2463eb; } .red { background: #d33f49; }
.skip { background: #8795a1; } #hint { color: #52606d; margin-top: 12px; }
#gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; margin-top: 24px; }
.card { border: 1px solid #ccd6dd; padding: 8px; } .card img { width: 100%; display: block; }
.card-label { font-weight: bold; margin: 7px 0; } .card button { font-size: 12px; padding: 7px 8px; margin: 2px; }
.remove { background: #9b1c31; }
</style>
</head>
<body>
<h1>Silent disco training points</h1>
<div id="status">Loading...</div>
<div id="image-wrap"><img id="frame" alt="Event image. Click a headset or object to label it."><div id="selection"></div></div>
<div id="buttons">
  <button class="negative" data-label="negative">No headset</button>
  <button class="green" data-label="green">DJ-green</button>
  <button class="blue" data-label="blue">DJ-blue</button>
  <button class="red" data-label="red">DJ-red</button>
  <button class="skip" data-label="skip">Next image</button>
    <button id="auto-review" class="skip" type="button">Review auto points</button>
</div>
<div id="zoom-panel"><div>Selected 255x255 crop</div><img id="zoom" alt="Enlarged selected crop"><div id="prediction-note"></div></div>
<div id="hint">Click a point in the image, then choose its label. Label multiple points before selecting Next image.</div>
<h2>Saved training crops</h2>
<div id="gallery"></div>
<script>
let selected = null;
let currentSource = '';
let reviewPoints = [];
let reviewIndex = 0;
let reviewMode = false;
const frame = document.getElementById('frame');
const imageWrap = document.getElementById('image-wrap');
let selection = document.getElementById('selection');
const zoomPanel = document.getElementById('zoom-panel');
const zoom = document.getElementById('zoom');
const predictionNote = document.getElementById('prediction-note');
const status = document.getElementById('status');
const gallery = document.getElementById('gallery');
const autoReviewButton = document.getElementById('auto-review');
async function refresh() {
  const response = await fetch('/frame');
  const data = await response.json();
    currentSource = data.name;
    frame.onload = () => data.points.forEach(point => drawBox(point.x, point.y, point.label));
  frame.src = 'data:image/jpeg;base64,' + data.image;
  frame.dataset.width = data.width; frame.dataset.height = data.height;
    frame.dataset.cropWidth = data.crop_width; frame.dataset.cropHeight = data.crop_height;
    document.querySelectorAll('.selection').forEach(box => box.remove());
    selection = document.createElement('div'); selection.id = 'selection'; selection.className = 'selection'; imageWrap.appendChild(selection);
    zoomPanel.style.display = 'none'; zoom.removeAttribute('src'); predictionNote.textContent = '';
  selected = null;
  status.textContent = `${data.index + 1}/${data.total}  ${data.name}  |  saved crops: ${data.saved}`;
  document.querySelectorAll('button').forEach(button => button.disabled = false);
    await refreshGallery(currentSource);
}
function drawBox(imageX, imageY, label) {
    const imageBox = frame.getBoundingClientRect();
    const wrapBox = imageWrap.getBoundingClientRect();
    const displayScaleX = frame.dataset.width / frame.clientWidth;
    const displayScaleY = frame.dataset.height / frame.clientHeight;
    const cropWidth = Number(frame.dataset.cropWidth) / displayScaleX;
    const cropHeight = Number(frame.dataset.cropHeight) / displayScaleY;
    const box = document.createElement('div');
    box.className = `selection ${label}`;
    box.dataset.labeled = 'true';
    box.style.width = `${cropWidth}px`; box.style.height = `${cropHeight}px`;
    box.style.left = `${imageBox.left - wrapBox.left + frame.clientLeft + imageX / displayScaleX - cropWidth / 2}px`;
    box.style.top = `${imageBox.top - wrapBox.top + frame.clientTop + imageY / displayScaleY - cropHeight / 2}px`;
    box.style.display = 'block';
    imageWrap.appendChild(box);
}
async function refreshGallery(source) {
    const response = await fetch('/annotations');
    const annotations = (await response.json()).filter(item => item.source === source);
    gallery.innerHTML = annotations.map(item => `<div class="card">
        <img src="/crop?id=${item.id}" alt="${item.label} training crop">
        <div class="card-label">${item.label}</div>
        <button class="green" onclick="editCrop(${item.id}, 'green')">DJ-green</button>
        <button class="blue" onclick="editCrop(${item.id}, 'blue')">DJ-blue</button>
        <button class="red" onclick="editCrop(${item.id}, 'red')">DJ-red</button>
        <button class="negative" onclick="editCrop(${item.id}, 'negative')">No headset</button>
        <button class="remove" onclick="removeCrop(${item.id})">Remove</button>
    </div>`).join('');
}
async function startAutoReview() {
    autoReviewButton.disabled = true;
    const response = await fetch('/auto-points', { method: 'POST' });
    reviewPoints = await response.json();
    if (!reviewPoints.length) {
        reviewMode = false;
        autoReviewButton.disabled = false;
        status.textContent = 'No new candidate points were found for this image.';
        return;
    }
    reviewMode = true;
    reviewIndex = 0;
    document.querySelector('[data-label="skip"]').disabled = true;
    showReviewPoint();
}
function showReviewPoint() {
    const point = reviewPoints[reviewIndex];
    selectPoint(point.x, point.y);
    const presence = point.predicted_presence ? `${point.predicted_presence}${point.presence_confidence == null ? '' : ` (${Math.round(point.presence_confidence * 100)}%)`}` : 'unavailable';
    const color = point.predicted_color ? `${point.predicted_color}${point.color_confidence == null ? '' : ` (${Math.round(point.color_confidence * 100)}%)`}` : 'unavailable';
    predictionNote.textContent = `AI guess: ${presence}, color ${color}`;
    status.textContent = `Auto-review ${reviewIndex + 1}/${reviewPoints.length}: choose the true label.`;
}
autoReviewButton.addEventListener('click', startAutoReview);
async function editCrop(id, label) {
    await fetch('/edit', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id, label}) });
    await refresh();
}
async function removeCrop(id) {
    await fetch('/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id}) });
    await refresh();
}
function selectPoint(displayX, displayY) {
    if (selection.dataset.labeled === 'true') {
        const nextSelection = document.createElement('div');
        nextSelection.className = 'selection';
        imageWrap.appendChild(nextSelection);
        selection = nextSelection;
    }
    const imageBox = frame.getBoundingClientRect();
    const wrapBox = imageWrap.getBoundingClientRect();
    const displayScaleX = frame.dataset.width / frame.clientWidth;
    const displayScaleY = frame.dataset.height / frame.clientHeight;
    selected = { x: displayX, y: displayY };
    const cropWidth = Number(frame.dataset.cropWidth) / displayScaleX;
    const cropHeight = Number(frame.dataset.cropHeight) / displayScaleY;
    selection.style.width = `${cropWidth}px`;
    selection.style.height = `${cropHeight}px`;
    selection.style.left = `${imageBox.left - wrapBox.left + frame.clientLeft + displayX / displayScaleX - cropWidth / 2}px`;
    selection.style.top = `${imageBox.top - wrapBox.top + frame.clientTop + displayY / displayScaleY - cropHeight / 2}px`;
    selection.style.display = 'block'; selection.className = 'selection'; selection.dataset.labeled = 'false';
    zoom.src = `/preview?x=${selected.x}&y=${selected.y}&t=${Date.now()}`;
    predictionNote.textContent = '';
    zoomPanel.style.display = 'block';
}
frame.addEventListener('click', event => {
        const imageBox = frame.getBoundingClientRect();
        const imageX = event.clientX - imageBox.left - frame.clientLeft;
        const imageY = event.clientY - imageBox.top - frame.clientTop;
        const displayScaleX = frame.dataset.width / frame.clientWidth;
        const displayScaleY = frame.dataset.height / frame.clientHeight;
        selectPoint(Math.round(imageX * displayScaleX), Math.round(imageY * displayScaleY));
        status.textContent += `  |  point selected (${selected.x}, ${selected.y})`;
});
document.querySelectorAll('button').forEach(button => button.addEventListener('click', async () => {
  const label = button.dataset.label;
  if (label !== 'skip' && !selected) { status.textContent = 'Select a point in the image first.'; return; }
  document.querySelectorAll('button').forEach(item => item.disabled = true);
  await fetch('/label', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ label, x: selected ? selected.x : 0, y: selected ? selected.y : 0 }) });
    if (label === 'skip') {
        await refresh();
    } else if (reviewMode) {
        selection.className = `selection ${label}`;
        selection.dataset.labeled = 'true';
        await refreshGallery(currentSource);
        reviewIndex += 1;
        if (reviewIndex < reviewPoints.length) {
            document.querySelectorAll('button').forEach(item => item.disabled = false);
            document.querySelector('[data-label="skip"]').disabled = true;
            showReviewPoint();
        } else {
            reviewMode = false;
            await refresh();
        }
    } else {
        selection.className = `selection ${label}`;
        selection.dataset.labeled = 'true';
        status.textContent += `  |  saved as ${label}`;
        selected = null;
        document.querySelectorAll('button').forEach(item => item.disabled = false);
        await refreshGallery(currentSource);
    }
}));
refresh();
</script>
</body></html>"""


class AnnotationApp:
    def __init__(self, image_dir: Path, dataset_dir: Path, detector_model: Path, presence_model: Path | None, color_model: Path | None, predictor_device: str, detector_image_size: int = 1280, detector_tile_size: int = 3000, max_side: int = 1600) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)
        self.videos = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
        if not self.images and not self.videos:
            raise ValueError(f"No images or videos found in {image_dir}")
        self.dataset_dir = dataset_dir
        self.max_side = max_side
        self.detector_image_size = detector_image_size
        self.detector_tile_size = detector_tile_size
        self.detector = load_person_detector(detector_model, torch.device("cpu"))
        self.predictor_device = resolve_device(predictor_device)
        self.presence_model = load_model(presence_model, 2, self.predictor_device)
        self.color_model = load_model(color_model, len(COLOR_MODEL_CLASSES), self.predictor_device)
        self.sequence_index = 0
        self.current_path: Path | None = None
        self.current_image = None
        self.current_source = ""
        self.saved = 0
        self.lock = threading.Lock()
        self.manifest_path = self.dataset_dir / "annotations.json"
        self.annotations = self.load_annotations()
        self.saved = len(self.annotations)
        self.next_id = max((int(item["id"]) for item in self.annotations), default=-1) + 1
        self.advance_media()

    def load_annotations(self) -> list[dict[str, object]]:
        if not self.manifest_path.exists():
            return []
        return json.loads(self.manifest_path.read_text())

    def write_annotations(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.annotations, indent=2))

    def advance_media(self) -> None:
        use_video = self.sequence_index % 2 == 1 and bool(self.videos)
        if use_video:
            path = random.choice(self.videos)
            capture = cv2.VideoCapture(str(path))
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            frame_number = random.randrange(frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f"Could not read random frame {frame_number} from {path}")
            self.current_source = f"{path.name}#frame={frame_number}"
        else:
            if not self.images:
                self.sequence_index += 1
                return self.advance_media()
            path = random.choice(self.images)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read {path}")
            self.current_source = path.name
        self.current_path = path
        self.current_image = image
        self.sequence_index += 1

    def current(self) -> tuple[Path, object, str]:
        if self.current_path is None or self.current_image is None:
            raise ValueError("No current media selected")
        return self.current_path, self.current_image, self.current_source

    def frame_payload(self) -> dict[str, object]:
        path, image, source = self.current()
        original_height, original_width = image.shape[:2]
        scale = min(1.0, self.max_side / max(original_height, original_width))
        display = cv2.resize(image, (round(original_width * scale), round(original_height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
        ok, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError(f"Could not encode {path}")
        points = [{"x": round(int(annotation["x"]) * scale), "y": round(int(annotation["y"]) * scale), "label": annotation["label"]} for annotation in self.annotations if annotation["source"] == source]
        return {"image": base64.b64encode(encoded).decode("ascii"), "width": display.shape[1], "height": display.shape[0], "crop_width": round(255 * scale), "crop_height": round(255 * scale), "name": source, "index": self.sequence_index - 1, "total": len(self.images) + len(self.videos), "saved": self.saved, "points": points}

    def preview(self, display_x: int, display_y: int) -> bytes:
        _, image, _ = self.current()
        original_height, original_width = image.shape[:2]
        scale = min(1.0, self.max_side / max(original_height, original_width))
        crop = crop_255(image, round(display_x / scale), round(display_y / scale))
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Could not encode preview crop")
        return encoded.tobytes()

    def auto_points(self) -> list[dict[str, int | float | str | None]]:
        _, image, source = self.current()
        people = detect_people(self.detector, image, confidence=0.03, image_size=self.detector_image_size, tile_size=self.detector_tile_size)
        display_scale = min(1.0, self.max_side / max(image.shape[:2]))
        existing = [(int(item["x"]), int(item["y"])) for item in self.annotations if item["source"] == source]
        selected = list(existing)
        points = []
        crops = []
        point_data = []
        for box, score in people:
            crop, (x, y) = crop_person_head(image, box)
            if any((x - old_x) ** 2 + (y - old_y) ** 2 < 60 ** 2 for old_x, old_y in selected):
                continue
            selected.append((x, y))
            crops.append(crop)
            point_data.append({"x": round(x * display_scale), "y": round(y * display_scale), "score": score})
        presence_predictions = predict_probabilities(self.presence_model, crops, self.predictor_device) if self.presence_model is not None else None
        color_predictions = [None] * len(crops)
        if self.color_model is not None:
            color_indices = [index for index in range(len(crops)) if self.presence_model is None or float(presence_predictions[index, 1].item()) >= 0.5]
            if color_indices:
                predicted = predict_probabilities(self.color_model, [crops[index] for index in color_indices], self.predictor_device)
                for prediction_index, crop_index in enumerate(color_indices):
                    color_predictions[crop_index] = predicted[prediction_index]
        for index, point in enumerate(point_data):
            if presence_predictions is not None:
                presence_probability = float(presence_predictions[index, 1].item())
                if 0.4 <= presence_probability <= 0.6:
                    point["predicted_presence"] = "uncertain"
                else:
                    point["predicted_presence"] = "headset" if presence_probability >= 0.5 else "no headset"
                point["presence_confidence"] = round(max(presence_probability, 1.0 - presence_probability), 3)
            else:
                point["predicted_presence"] = None
                point["presence_confidence"] = None
            if color_predictions[index] is not None:
                color_index = int(color_predictions[index].argmax().item())
                point["predicted_color"] = COLOR_MODEL_CLASSES[color_index]
                point["color_confidence"] = round(float(color_predictions[index][color_index].item()), 3)
            else:
                point["predicted_color"] = "not evaluated"
                point["color_confidence"] = None
            points.append(point)
        return points

    def save(self, label: str, display_x: int, display_y: int) -> None:
        if label == "skip":
            self.advance_media()
            return
        path, image, source = self.current()
        original_height, original_width = image.shape[:2]
        scale = min(1.0, self.max_side / max(original_height, original_width))
        x = round(display_x / scale)
        y = round(display_y / scale)
        crop = crop_255(image, x, y)
        presence_label = "negative" if label == "negative" else "headset"
        color_label = "unknown" if label == "negative" else label
        filename = f"{path.stem}_{self.next_id:06d}.jpg"
        for task, category in (("presence", presence_label), ("color", color_label)):
            destination = self.dataset_dir / task / category
            destination.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination / filename), crop):
                raise ValueError(f"Could not save {destination / filename}")
        self.annotations.append({"id": self.next_id, "label": label, "filename": filename, "source": source, "x": x, "y": y})
        self.next_id += 1
        self.saved += 1
        self.write_annotations()

    def edit(self, annotation_id: int, label: str) -> None:
        annotation = next(item for item in self.annotations if item["id"] == annotation_id)
        old_label = str(annotation["label"])
        if label == old_label:
            return
        filename = str(annotation["filename"])
        old_presence = "negative" if old_label == "negative" else "headset"
        old_color = "unknown" if old_label == "negative" else old_label
        new_presence = "negative" if label == "negative" else "headset"
        new_color = "unknown" if label == "negative" else label
        moves = (("presence", old_presence, new_presence), ("color", old_color, new_color))
        for task, old_category, new_category in moves:
            old_path = self.dataset_dir / task / old_category / filename
            new_path = self.dataset_dir / task / new_category / filename
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
        annotation["label"] = label
        self.write_annotations()

    def delete(self, annotation_id: int) -> None:
        annotation = next(item for item in self.annotations if item["id"] == annotation_id)
        filename = str(annotation["filename"])
        label = str(annotation["label"])
        presence = "negative" if label == "negative" else "headset"
        color = "unknown" if label == "negative" else label
        for task, category in (("presence", presence), ("color", color)):
            path = self.dataset_dir / task / category / filename
            if path.exists():
                path.unlink()
        self.annotations.remove(annotation)
        self.write_annotations()


class Handler(BaseHTTPRequestHandler):
    app: AnnotationApp

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/frame":
            self.respond(200, "application/json", json.dumps(self.app.frame_payload()).encode())
        elif urlparse(self.path).path == "/annotations":
            self.respond(200, "application/json", json.dumps(self.app.annotations).encode())
        elif urlparse(self.path).path == "/crop":
            query = urlparse(self.path).query
            annotation_id = int(query.split("=", 1)[1])
            annotation = next(item for item in self.app.annotations if item["id"] == annotation_id)
            label = str(annotation["label"])
            color = "unknown" if label == "negative" else label
            path = self.app.dataset_dir / "color" / color / str(annotation["filename"])
            ok, encoded = cv2.imencode(".jpg", cv2.imread(str(path)))
            if not ok:
                self.respond(404, "text/plain", b"Crop not found")
            else:
                self.respond(200, "image/jpeg", encoded.tobytes())
        elif urlparse(self.path).path == "/preview":
            query = parse_qs(urlparse(self.path).query)
            preview = self.app.preview(int(query["x"][0]), int(query["y"][0]))
            self.respond(200, "image/jpeg", preview)
        else:
            self.respond(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/auto-points":
            with self.app.lock:
                points = self.app.auto_points()
            self.respond(200, "application/json", json.dumps(points).encode())
            return
        if route not in {"/label", "/edit", "/delete"}:
            self.respond(404, "text/plain", b"Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.app.lock:
            if route == "/label":
                self.app.save(str(payload["label"]), int(payload["x"]), int(payload["y"]))
            elif route == "/edit":
                self.app.edit(int(payload["id"]), str(payload["label"]))
            else:
                self.app.delete(int(payload["id"]))
        self.respond(204, "", b"")

    def respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Label headset points and save 255x255 training crops.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--presence-model", type=Path, default=Path("models/presence.pt"))
    parser.add_argument("--color-model", type=Path, default=Path("models/color.pt"))
    parser.add_argument("--predictor-device", choices=("auto", "cpu", "cuda", "rocm", "xpu"), default="cpu")
    parser.add_argument("--detector-image-size", type=int, default=1280)
    parser.add_argument("--detector-tile-size", type=int, default=3000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the annotation page in the default browser.")
    args = parser.parse_args()
    app = AnnotationApp(args.image_dir, args.dataset, args.detector_model, args.presence_model, args.color_model, args.predictor_device, args.detector_image_size, args.detector_tile_size)
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Open {url} to annotate. Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping annotator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
