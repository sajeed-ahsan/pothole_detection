import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = os.getenv("POTHOLE_MODEL", "runs/train/pothole_detection_run12/weights/best.pt")
DEFAULT_SOURCE = os.getenv("POTHOLE_SOURCE", "")
CONF_THRESHOLD = float(os.getenv("POTHOLE_CONF", "0.35"))
IOU_THRESHOLD = float(os.getenv("POTHOLE_IOU", "0.5"))
MIN_AREA = float(os.getenv("POTHOLE_MIN_AREA", "0.0015"))
HIGH_THRESHOLD = float(os.getenv("POTHOLE_HIGH_AREA", "0.02"))
MIN_ASPECT_RATIO = float(os.getenv("POTHOLE_MIN_AR", "0.25"))
MAX_ASPECT_RATIO = float(os.getenv("POTHOLE_MAX_AR", "4.0"))
BASE_LAT = float(os.getenv("BASE_LAT", "12.9716"))
BASE_LNG = float(os.getenv("BASE_LNG", "77.5946"))
DB_PATH = os.getenv("POTHOLE_DB_PATH", "pothole_detections.db")

MODEL: Optional[YOLO] = None
MODEL_LOAD_ERROR = None
STATE_LOCK = threading.Lock()

VIDEO_CAPTURE: Optional[cv2.VideoCapture] = None
VIDEO_SOURCE: str = ""
TRACKED_POTHOLES: List[Dict] = []
NEXT_TRACK_ID = 1
TRACK_MATCH_DEG = float(os.getenv("TRACK_MATCH_DEG", "0.00035"))
TRACK_MAX_MISSES = int(os.getenv("TRACK_MAX_MISSES", "8"))


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pothole_tracks (
                track_id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                severity TEXT NOT NULL,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def load_model() -> None:
    global MODEL, MODEL_LOAD_ERROR
    try:
        if Path(MODEL_PATH).exists():
            MODEL = YOLO(MODEL_PATH)
        else:
            MODEL_LOAD_ERROR = f"Model not found: {MODEL_PATH}"
    except Exception as exc:  # pragma: no cover
        MODEL_LOAD_ERROR = str(exc)


def classify_severity(relative_area: float) -> str:
    return "HIGH" if relative_area >= HIGH_THRESHOLD else "MEDIUM"


def relative_area_from_mask_or_box(
    frame_h: int,
    frame_w: int,
    box_xyxy: np.ndarray,
    mask: Optional[np.ndarray],
) -> float:
    frame_area = float(frame_h * frame_w)
    if mask is not None:
        return float(np.count_nonzero(mask > 0.5)) / frame_area

    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / frame_area


def aspect_ratio(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return w / h


def detect_potholes(frame: np.ndarray) -> List[Dict]:
    if MODEL is None:
        return []

    h, w = frame.shape[:2]
    results = MODEL.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    masks = None
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.cpu().numpy()

    filtered = []
    for idx, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        ar = aspect_ratio(xyxy)
        if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
            continue

        mask = None
        if masks is not None and idx < len(masks):
            mask = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)

        rel_area = relative_area_from_mask_or_box(h, w, xyxy, mask)
        if rel_area < MIN_AREA:
            continue

        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        filtered.append(
            {
                "bbox": [x1, y1, x2, y2],
                "severity": classify_severity(rel_area),
            }
        )

    return filtered


def bbox_to_gps(bbox: List[int], frame_w: int, frame_h: int, index: int) -> Dict:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x_norm = (cx / max(1.0, float(frame_w))) - 0.5
    y_norm = (cy / max(1.0, float(frame_h))) - 0.5

    lat = BASE_LAT + (y_norm * 0.0040) + (index * 0.00012)
    lng = BASE_LNG + (x_norm * 0.0060) + (index * 0.00012)
    return {"lat": round(lat, 6), "lng": round(lng, 6)}


def simulated_payload() -> List[Dict]:
    return [
        {"lat": 12.9716, "lng": 77.5946, "severity": "HIGH"},
        {"lat": 12.9722, "lng": 77.5954, "severity": "MEDIUM"},
    ]


def severity_rank(severity: str) -> int:
    return 2 if severity == "HIGH" else 1


def get_frame_from_source(source: str) -> Optional[np.ndarray]:
    global VIDEO_CAPTURE, VIDEO_SOURCE
    ext = Path(source).suffix.lower()
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    if not is_video:
        return cv2.imread(source)

    if VIDEO_CAPTURE is None or VIDEO_SOURCE != source:
        if VIDEO_CAPTURE is not None:
            VIDEO_CAPTURE.release()
        VIDEO_CAPTURE = cv2.VideoCapture(source)
        VIDEO_SOURCE = source

    if VIDEO_CAPTURE is None or not VIDEO_CAPTURE.isOpened():
        return None

    ok, frame = VIDEO_CAPTURE.read()
    if ok:
        return frame

    VIDEO_CAPTURE.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = VIDEO_CAPTURE.read()
    return frame if ok else None


def merge_with_tracker(detections: List[Dict], frame_w: int, frame_h: int) -> List[Dict]:
    global NEXT_TRACK_ID

    matched_ids = set()
    current_points = []
    for idx, det in enumerate(detections):
        gps = bbox_to_gps(det["bbox"], frame_w, frame_h, idx)
        current_points.append({"lat": gps["lat"], "lng": gps["lng"], "severity": det["severity"]})

    for point in current_points:
        best = None
        best_dist = float("inf")
        for track in TRACKED_POTHOLES:
            dist = ((point["lat"] - track["lat"]) ** 2 + (point["lng"] - track["lng"]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = track

        if best is not None and best_dist <= TRACK_MATCH_DEG:
            best["lat"] = round((best["lat"] * 0.7) + (point["lat"] * 0.3), 6)
            best["lng"] = round((best["lng"] * 0.7) + (point["lng"] * 0.3), 6)
            if severity_rank(point["severity"]) > severity_rank(best["severity"]):
                best["severity"] = point["severity"]
            best["misses"] = 0
            matched_ids.add(best["id"])
        else:
            TRACKED_POTHOLES.append(
                {
                    "id": NEXT_TRACK_ID,
                    "lat": point["lat"],
                    "lng": point["lng"],
                    "severity": point["severity"],
                    "misses": 0,
                }
            )
            matched_ids.add(NEXT_TRACK_ID)
            NEXT_TRACK_ID += 1

    for track in TRACKED_POTHOLES:
        if track["id"] not in matched_ids:
            track["misses"] += 1

    TRACKED_POTHOLES[:] = [t for t in TRACKED_POTHOLES if t["misses"] <= TRACK_MAX_MISSES]

    return [{"lat": t["lat"], "lng": t["lng"], "severity": t["severity"]} for t in TRACKED_POTHOLES]


def persist_tracks(source: str) -> None:
    if not TRACKED_POTHOLES:
        return

    rows = [(t["id"], source, t["lat"], t["lng"], t["severity"]) for t in TRACKED_POTHOLES]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO pothole_tracks (track_id, source, lat, lng, severity, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(track_id) DO UPDATE SET
                source = excluded.source,
                lat = excluded.lat,
                lng = excluded.lng,
                severity = CASE
                    WHEN excluded.severity = 'HIGH' THEN 'HIGH'
                    ELSE pothole_tracks.severity
                END,
                last_seen = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            rows,
        )
        conn.commit()


@app.route("/")
def home():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/potholes", methods=["GET"])
def api_potholes():
    source = request.args.get("source", DEFAULT_SOURCE).strip()
    reset = request.args.get("reset", "0").lower() in {"1", "true", "yes"}

    with STATE_LOCK:
        if reset:
            TRACKED_POTHOLES.clear()

        if not source:
            return jsonify(simulated_payload())

        frame = get_frame_from_source(source)
        if frame is None:
            return jsonify({"error": f"Could not read source image/video: {source}"}), 400

        detections = detect_potholes(frame)
        h, w = frame.shape[:2]
        response = merge_with_tracker(detections, w, h)
        persist_tracks(source)
        return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": MODEL_PATH,
            "model_loaded": MODEL is not None,
            "model_error": MODEL_LOAD_ERROR,
            "tracked_potholes": len(TRACKED_POTHOLES),
            "video_source": VIDEO_SOURCE,
            "db_path": str(Path(DB_PATH).resolve()),
        }
    )


@app.route("/api/potholes/history", methods=["GET"])
def pothole_history():
    limit = max(1, min(int(request.args.get("limit", "100")), 1000))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT track_id, source, lat, lng, severity, first_seen, last_seen
            FROM pothole_tracks
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return jsonify([dict(r) for r in rows])


load_model()
init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
