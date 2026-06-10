from ultralytics import YOLO
import cv2
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

VIDEO_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = str(VIDEO_DIR / "runs" / "parking_detector" / "parksight_yolov8s" / "weights" / "best.pt")
VIDEO_PATH = str(VIDEO_DIR / "parking_video.mp4")

CONFIDENCE = 0.5

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

print("Classes:", model.names)

# Expected:
# {0: 'empty', 1: 'occupied'}

# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Failed to open video.")
    exit()

# ==========================================
# OUTPUT VIDEO
# ==========================================

out_width = 1280
out_height = 720

fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "parking_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (out_width, out_height)
)

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for faster inference
    frame = cv2.resize(frame, (out_width, out_height))

    # Detection
    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated_frame = frame.copy()

    occupied = 0
    empty = 0

    # ======================================
    # DRAW BOXES
    # ======================================

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[int(cls)].lower()

            # EMPTY -> GREEN

            if class_name == "empty":

                empty += 1

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # OCCUPIED -> RED

            elif class_name == "occupied":

                occupied += 1

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

    # ======================================
    # CALCULATE OCCUPANCY
    # ======================================

    total_slots = occupied + empty

    occupancy_rate = (
        occupied / total_slots * 100
        if total_slots > 0
        else 0
    )

    # ======================================
    # PROFESSIONAL HUD
    # ======================================

    overlay = annotated_frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (300, 160),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        annotated_frame,
        0.45,
        0,
        annotated_frame
    )

    # Title

    cv2.putText(
        annotated_frame,
        "PARKING STATUS",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Occupied

    cv2.putText(
        annotated_frame,
        f"Occupied : {occupied}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    # Empty

    cv2.putText(
        annotated_frame,
        f"Empty     : {empty}",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    # Occupancy %

    cv2.putText(
        annotated_frame,
        f"Occupancy : {occupancy_rate:.1f}%",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    # ======================================
    # DISPLAY & SAVE
    # ======================================

    cv2.imshow(
        "AI Parking Occupancy Tracking",
        annotated_frame
    )

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nProcessing Finished")
print("Output saved as parking_output.mp4")