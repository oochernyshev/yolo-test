# yolo-test

A simple application for testing video analytics with YOLO models.

## Features

- Run object detection or multi-object tracking on any video file or webcam stream
- Annotated output with bounding boxes, class labels, and confidence scores
- Live FPS and detection-count overlay
- Per-class detection statistics summary printed after processing
- Optional saving of the annotated video

## Requirements

- Python ≥ 3.10
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```
python app.py [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--source PATH\|INDEX` | `0` | Video file path or camera index |
| `--model FILE` | `yolov8n.pt` | YOLO model weights (downloaded automatically on first use) |
| `--output FILE` | *(none)* | Save annotated video to this path |
| `--conf FLOAT` | `0.25` | Confidence threshold |
| `--iou FLOAT` | `0.45` | IoU threshold for NMS |
| `--track` | off | Enable multi-object tracking |
| `--show` | off | Display a live preview window (press **q** to quit) |
| `--device STR` | auto | Inference device (`cpu`, `0` for GPU, …) |
| `--classes INT…` | *(all)* | Only detect these class indices |

### Examples

```bash
# Webcam with live preview
python app.py --show

# Analyze a video file and save annotated output
python app.py --source video.mp4 --output out.mp4

# Use a larger model, enable tracking, filter to persons (class 0)
python app.py --source video.mp4 --model yolov8m.pt --track --classes 0 --show

# Run on GPU
python app.py --source video.mp4 --device 0
```

## Running tests

```bash
pip install pytest
pytest test_app.py -v
```
