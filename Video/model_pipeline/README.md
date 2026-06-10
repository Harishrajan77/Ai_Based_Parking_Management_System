# ParkSight Occupancy Model Pipeline

Everything in this pipeline stays inside the `Video` folder:

- source images: `Video/images`
- source YOLO labels: `Video/labels`
- split dataset: `Video/parking_occupancy_yolo`
- pretrained weights cache: `Video/pretrained`
- training outputs: `Video/runs`

## Prepare Train / Val / Test

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\prepare_yolo_dataset.py --overwrite
```

## Train Best GPU Model

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\train_parking_detector.py
```

If GPU memory is low:

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\train_parking_detector.py --batch 4
```

## Real-Time Occupancy

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\realtime_occupancy.py
```
# ParkSight Occupancy Model Pipeline

Everything in this pipeline stays inside the `Video` folder:

- source images: `Video/images`
- source YOLO labels: `Video/labels`
- split dataset: `Video/parking_occupancy_yolo`
- pretrained weights cache: `Video/pretrained`
- training outputs: `Video/runs`

## Prepare Train / Val / Test

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\prepare_yolo_dataset.py --overwrite
```

This creates:

```text
Video\parking_occupancy_yolo\train
Video\parking_occupancy_yolo\val
Video\parking_occupancy_yolo\test
```

## Train Best GPU Model

```powershell
python AI_Based_Parking_Management_System\Video\train_parking_detector.py
```

Default training settings:

- GPU required by default
- `yolov8s.pt`
- `100` epochs
- `960` image size
- batch `8`
- cosine learning rate
- AMP enabled
- checkpoints every `10` epochs
- early stopping patience `30`

If GPU memory is low:

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\train_parking_detector.py --batch 4
```

## Real-Time Occupancy

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\realtime_occupancy.py
```

Webcam:

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\realtime_occupancy.py --source 0
```

Save output video:

```powershell
python AI_Based_Parking_Management_System\Video\model_pipeline\realtime_occupancy.py --save AI_Based_Parking_Management_System\Video\runs\parksight_demo.mp4
```
