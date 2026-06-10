import argparse
import pickle
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = VIDEO_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_WEIGHTS = VIDEO_DIR / "runs" / "parking_detector" / "parksight_yolov8s" / "weights" / "best.pt"
DEFAULT_VIDEO = VIDEO_DIR / "parking_video.mp4"
DEFAULT_REGRESSION_MODEL = WORKSPACE_DIR / "DAILY_TASKS" / "day_04" / "task_06" / "parking_multiple_linear_regression_model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time parking occupancy detection.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO detector weights.")
    parser.add_argument("--source", default=str(DEFAULT_VIDEO), help="Video path, webcam id like 0, or stream URL.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--regression-model", type=Path, default=DEFAULT_REGRESSION_MODEL, help="Optional occupancy rate regression pickle.")
    parser.add_argument("--save", type=Path, default=None, help="Optional output video path.")
    parser.add_argument("--no-window", action="store_true", help="Run without showing an OpenCV window.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def install_sklearn_compat() -> None:
    if "sklearn.linear_model._base" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    linear_model_module = types.ModuleType("sklearn.linear_model")
    base_module = types.ModuleType("sklearn.linear_model._base")

    class LinearRegression:
        def predict(self, rows):
            predictions = []
            for row in rows:
                total = float(getattr(self, "intercept_", 0.0))
                total += sum(float(coef) * float(value) for coef, value in zip(getattr(self, "coef_", []), row))
                predictions.append(total)
            return predictions

    base_module.LinearRegression = LinearRegression
    linear_model_module.LinearRegression = LinearRegression
    linear_model_module._base = base_module
    sklearn_module.linear_model = linear_model_module

    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.linear_model"] = linear_model_module
    sys.modules["sklearn.linear_model._base"] = base_module


def load_regression_model(path: Path):
    path = resolve(path)
    if not path.exists():
        return None

    install_sklearn_compat()
    with open(path, "rb") as handle:
        return pickle.load(handle)


def open_source(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id == 1:
        return 36, 54, 220
    return 54, 168, 65


def draw_panel(frame, empty_count: int, occupied_count: int, detected_rate: float, predicted_rate: float | None, fps: float) -> None:
    panel_width = 430
    cv2.rectangle(frame, (18, 18), (panel_width, 178), (15, 28, 38), -1)
    cv2.rectangle(frame, (18, 18), (panel_width, 178), (220, 232, 240), 1)

    cv2.putText(frame, "ParkSight Live Occupancy", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupied slots: {occupied_count}", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 255), 2)
    cv2.putText(frame, f"Empty slots: {empty_count}", (34, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 255, 220), 2)
    cv2.putText(frame, f"Detected occupancy: {detected_rate:.2f}%", (34, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 244, 214), 2)

    model_text = "Forecast: unavailable" if predicted_rate is None else f"Forecast: {predicted_rate:.2f}%"
    cv2.putText(frame, f"{model_text}  |  FPS: {fps:.1f}", (34, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (196, 224, 255), 1)


def predict_rate(regression_model, empty_count: int, occupied_count: int, previous_occupied: int | None) -> float | None:
    if regression_model is None:
        return None

    now = datetime.now()
    total_slots = empty_count + occupied_count
    entry_count = 0 if previous_occupied is None else max(0, occupied_count - previous_occupied)
    exit_count = 0 if previous_occupied is None else max(0, previous_occupied - occupied_count)
    vehicle_count = total_slots

    features = [[now.hour, occupied_count, empty_count, vehicle_count, entry_count, exit_count]]
    return max(0.0, min(100.0, float(regression_model.predict(features)[0])))


def run_detection(args: argparse.Namespace) -> None:
    weights_path = resolve(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}. Train the detector first.")

    model = YOLO(str(weights_path))
    regression_model = load_regression_model(args.regression_model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        save_path = resolve(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    previous_occupied = None
    previous_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        empty_count = 0
        occupied_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
            label = "occupied" if class_id == 1 else "empty"
            color = color_for_class(class_id)

            if class_id == 1:
                occupied_count += 1
            else:
                empty_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

        total = empty_count + occupied_count
        detected_rate = 0.0 if total == 0 else (occupied_count / total) * 100
        predicted_rate = predict_rate(regression_model, empty_count, occupied_count, previous_occupied)

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        draw_panel(frame, empty_count, occupied_count, detected_rate, predicted_rate, fps)
        previous_occupied = occupied_count

        if writer:
            writer.write(frame)

        if not args.no_window:
            cv2.imshow("ParkSight Live Occupancy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_detection(args)


if __name__ == "__main__":
    main()
