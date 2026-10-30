import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def classify_severity(relative_area: float, high_threshold: float) -> str:
    """Return severity as HIGH or MEDIUM based on normalized pothole area."""
    return "HIGH" if relative_area >= high_threshold else "MEDIUM"


def mask_or_box_relative_area(
    image_h: int,
    image_w: int,
    box_xyxy: np.ndarray,
    mask: Optional[np.ndarray],
) -> float:
    """Compute normalized pothole area using mask if present, otherwise box area."""
    image_area = float(image_h * image_w)

    if mask is not None:
        mask_area = float(np.count_nonzero(mask > 0.5))
        return mask_area / image_area

    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / image_area


def box_aspect_ratio(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return w / h


def detect_potholes(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    min_area: float,
    high_threshold: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> List[Dict]:
    h, w = frame.shape[:2]
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    result = results[0]

    detections: List[Dict] = []
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return detections

    mask_data = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None

    for idx, box in enumerate(boxes):
        score = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy()
        ar = box_aspect_ratio(xyxy)
        if ar < min_aspect_ratio or ar > max_aspect_ratio:
            continue

        mask = None
        if mask_data is not None and idx < len(mask_data):
            mask = cv2.resize(mask_data[idx], (w, h), interpolation=cv2.INTER_NEAREST)

        rel_area = mask_or_box_relative_area(h, w, xyxy, mask)
        if rel_area < min_area:
            continue

        severity = classify_severity(rel_area, high_threshold)
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]

        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": round(score, 4),
                "relative_area": round(rel_area, 6),
                "severity": severity,
            }
        )

    return detections


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        sev = det["severity"]
        color = (0, 0, 255) if sev == "HIGH" else (0, 255, 255)  # Red / Yellow
        label = f"{sev} | conf={det['confidence']:.2f} | area={det['relative_area']:.4f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def run_on_image(args: argparse.Namespace, model: YOLO) -> None:
    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    detections = detect_potholes(
        model=model,
        frame=img,
        conf=args.conf,
        iou=args.iou,
        min_area=args.min_area,
        high_threshold=args.high_threshold,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
    )

    annotated = draw_detections(img, detections)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(args.source).name
    out_img = out_dir / f"annotated_{image_name}"
    out_json = out_dir / f"detections_{Path(image_name).stem}.json"

    cv2.imwrite(str(out_img), annotated)

    payload = {
        "source": str(Path(args.source).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_detections": len(detections),
        "detections": detections,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {out_img}")
    print(f"Saved: {out_json}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOv8 pothole inference with HIGH/MEDIUM severity")
    p.add_argument("--model", type=str, default="runs/train/pothole_yolov8_seg/weights/best.pt")
    p.add_argument("--source", type=str, required=True, help="Path to input image")
    p.add_argument("--output-dir", type=str, default="prediction_results_v2")

    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-area", type=float, default=0.0015, help="Ignore tiny detections")
    p.add_argument("--high-threshold", type=float, default=0.02, help="Relative area threshold for HIGH")
    p.add_argument("--min-aspect-ratio", type=float, default=0.25)
    p.add_argument("--max-aspect-ratio", type=float, default=4.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    run_on_image(args, model)


if __name__ == "__main__":
    main()
