import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

VIDEO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = VIDEO_DIR / "parking_occupancy_yolo" / "data.yaml"
DEFAULT_OUTPUT = VIDEO_DIR / "runs" / "parking_detector"
DEFAULT_PRETRAINED_DIR = VIDEO_DIR / "pretrained"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolov8s.pt")

    parser.add_argument("--epochs", type=int, default=150)

    # Faster and usually sufficient for parking lots
    parser.add_argument("--imgsz", type=int, default=640)

    # AutoBatch
    parser.add_argument("--batch", type=float, default=-1)

    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="parksight_yolov8s")

    # RTX4050 performs well with this
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--cache", action="store_true")

    return parser.parse_args()


def resolve(path: Path):
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def main():
    args = parse_args()

    data_path = resolve(args.data)
    project_path = resolve(args.project)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    DEFAULT_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(DEFAULT_PRETRAINED_DIR)

    try:
        model = YOLO(args.model)

        results = model.train(
            data=str(data_path),

            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,

            device=args.device,

            project=str(project_path),
            name=args.name,

            optimizer="AdamW",
            cos_lr=True,

            patience=50,

            cache=args.cache,
            workers=args.workers,

            amp=True,

            close_mosaic=15,

            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.6,

            translate=0.10,
            scale=0.50,

            fliplr=0.5,
            flipud=0.0,

            degrees=0.0,
            shear=0.0,
            perspective=0.0,

            plots=True,
            save=True,
            save_period=10,

            seed=42,
            exist_ok=True,
        )

    finally:
        os.chdir(old_cwd)

    print(results)


if __name__ == "__main__":
    main()