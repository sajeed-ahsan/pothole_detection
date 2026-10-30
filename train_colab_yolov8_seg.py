import os
from pathlib import Path

from ultralytics import YOLO


def ensure_roboflow_dataset(api_key: str, workspace: str, project: str, version: int, fmt: str = "yolov8-seg") -> Path:
    """Download dataset from Roboflow and return local dataset root."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "roboflow is not installed. In Colab run: pip install -q roboflow"
        ) from exc

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(fmt)
    return Path(ds.location)


def main() -> None:
    """
    Colab usage:
      1) pip install -q ultralytics roboflow
      2) Set env vars:
         - ROBOFLOW_API_KEY
         - ROBOFLOW_WORKSPACE
         - ROBOFLOW_PROJECT
         - ROBOFLOW_VERSION
      3) python train_colab_yolov8_seg.py
    """
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "")
    project = os.getenv("ROBOFLOW_PROJECT", "")
    version = int(os.getenv("ROBOFLOW_VERSION", "0"))

    if not api_key or not workspace or not project or version <= 0:
        raise ValueError(
            "Missing Roboflow config. Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, "
            "ROBOFLOW_PROJECT, and ROBOFLOW_VERSION env vars."
        )

    dataset_root = ensure_roboflow_dataset(
        api_key=api_key,
        workspace=workspace,
        project=project,
        version=version,
        fmt="yolov8-seg",
    )
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find data.yaml at: {data_yaml}")

    model = YOLO("yolov8s-seg.pt")

    # Settings tuned for reducing false positives on road-surface defects.
    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=960,
        batch=16,
        patience=30,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        degrees=3,
        translate=0.08,
        scale=0.25,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        mask_ratio=4,
        overlap_mask=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        amp=True,
        workers=2,
        project="runs/train",
        name="pothole_yolov8_seg",
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True,
    )

    best = Path("runs/train/pothole_yolov8_seg/weights/best.pt")
    print(f"Training complete. Best model: {best.resolve()}")


if __name__ == "__main__":
    main()
