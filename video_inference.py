import cv2
import streamlit as st
from ultralytics import YOLO


_OVERLAY_PALETTE_BGR = [
    (62, 179, 255),
    (0, 210, 140),
    (120, 90, 255),
    (255, 170, 0),
    (255, 120, 160),
    (130, 220, 60),
]


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _overlay_detected_frames(image_bgr, boxes, alpha: float = 0.22):
    if boxes is None or not hasattr(boxes, "xyxy") or boxes.xyxy is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    painted = []

    n = len(boxes.xyxy)
    for i in range(n):
        try:
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        except Exception:
            continue

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color_key = i
        if hasattr(boxes, "id") and boxes.id is not None and i < len(boxes.id):
            color_key = _safe_int(boxes.id[i].item(), i)
        elif hasattr(boxes, "cls") and boxes.cls is not None and i < len(boxes.cls):
            color_key = _safe_int(boxes.cls[i].item(), i)

        color = _OVERLAY_PALETTE_BGR[color_key % len(_OVERLAY_PALETTE_BGR)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        painted.append(((x1, y1), (x2, y2), color))

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)
    for (p1, p2, color) in painted:
        cv2.rectangle(blended, p1, p2, color, 2)

    return blended


def _collect_detections(result, include_keypoints: bool = False):
    if result is None or result.boxes is None or not hasattr(result.boxes, "xyxy"):
        return []

    boxes = result.boxes
    names = result.names if hasattr(result, "names") else {}
    keypoints_obj = result.keypoints if include_keypoints and hasattr(result, "keypoints") else None
    keypoints_xy = keypoints_obj.xy if keypoints_obj is not None and hasattr(keypoints_obj, "xy") else None

    detections = []
    n = len(boxes.xyxy)
    for i in range(n):
        try:
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        except Exception:
            continue

        det = {
            "bbox_xyxy": [x1, y1, x2, y2],
        }

        if hasattr(boxes, "conf") and boxes.conf is not None and i < len(boxes.conf):
            conf_val = _safe_float(boxes.conf[i].item())
            if conf_val is not None:
                det["confidence"] = conf_val

        class_id = None
        if hasattr(boxes, "cls") and boxes.cls is not None and i < len(boxes.cls):
            class_id = _safe_int(boxes.cls[i].item(), default=None)
            if class_id is not None:
                det["class_id"] = class_id
                if isinstance(names, dict) and class_id in names:
                    det["class_name"] = str(names[class_id])

        if hasattr(boxes, "id") and boxes.id is not None and i < len(boxes.id):
            track_id = _safe_int(boxes.id[i].item(), default=None)
            if track_id is not None:
                det["track_id"] = track_id

        if keypoints_xy is not None and i < len(keypoints_xy):
            try:
                det["keypoints_xy"] = [[float(px), float(py)] for px, py in keypoints_xy[i].tolist()]
            except Exception:
                pass

        detections.append(det)

    return detections


@st.cache_resource
def load_model(name: str):
    return YOLO(name)


def draw_people(
    model,
    frame_bgr,
    frame_box,
    conf: float,
    iou: float,
    imgsz: int,
    tracker: str,
    state,
):
    results = model.track(
        frame_bgr,
        classes=[0],
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        persist=True,
        tracker=tracker,
        device="mps",
        verbose=False,
    )
    n_people = len(results[0].boxes) if results and results[0].boxes is not None else 0
    detections = _collect_detections(results[0], include_keypoints=False) if results else []
    annotated_bgr = results[0].plot() if results else frame_bgr
    if results and results[0].boxes is not None:
        annotated_bgr = _overlay_detected_frames(annotated_bgr, results[0].boxes)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    frame_box.image(annotated_rgb, channels="RGB", width="stretch")
    state["last_frame_rgb"] = annotated_rgb
    state["last_people"] = n_people
    state["last_detections"] = detections
    return n_people


def draw_pose(
    model,
    frame_bgr,
    frame_box,
    pose_conf: float,
    pose_imgsz: int,
    pose_tracker: str,
    state,
):
    results = model.track(
        frame_bgr,
        conf=float(pose_conf),
        imgsz=int(pose_imgsz),
        persist=True,
        tracker=pose_tracker,
        device="mps",
        verbose=False,
    )

    n_people = 0
    if results and hasattr(results[0], "keypoints") and results[0].keypoints is not None:
        try:
            n_people = len(results[0].keypoints)
        except Exception:
            n_people = 0

    detections = _collect_detections(results[0], include_keypoints=True) if results else []

    annotated_bgr = results[0].plot() if results else frame_bgr
    if results and results[0].boxes is not None:
        annotated_bgr = _overlay_detected_frames(annotated_bgr, results[0].boxes)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    frame_box.image(annotated_rgb, channels="RGB", width="stretch")
    state["last_frame_rgb"] = annotated_rgb
    state["last_people"] = n_people
    state["last_detections"] = detections
    return n_people
