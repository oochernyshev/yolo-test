import cv2
import streamlit as st
from ultralytics import YOLO


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
    annotated_bgr = results[0].plot() if results else frame_bgr
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    frame_box.image(annotated_rgb, channels="RGB", width="stretch")
    state["last_frame_rgb"] = annotated_rgb
    state["last_people"] = n_people
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

    annotated_bgr = results[0].plot() if results else frame_bgr
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    frame_box.image(annotated_rgb, channels="RGB", width="stretch")
    state["last_frame_rgb"] = annotated_rgb
    state["last_people"] = n_people
    return n_people
