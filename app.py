import time
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

# Smooth Streamlit player (no per-frame st.rerun)
# - People tracking: YOLO26 (n/s/m)
# - Optional pose tracking (pretrained pose models)
# - Play / Pause / Stop
# - Adaptive real-time skipping when inference is slow

st.set_page_config(page_title="Video Analytics • YOLO26 + Pose", page_icon="🎥", layout="wide")
st.title("🎥 Video Analytics: People Tracking + Pose")

with st.sidebar:
    st.markdown("## Video")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])

    st.markdown("## Mode")
    mode = st.selectbox("Analytics", ["People tracking (YOLO26)", "Pose tracking"], index=0)

    if mode == "People tracking (YOLO26)":
        st.markdown("## People model (YOLO26)")
        people_model_name = st.selectbox("Model", ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"], index=1)
        tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

        st.markdown("### Detection")
        conf = st.slider("Confidence", 0.05, 0.90, 0.15, 0.05)
        iou = st.slider("IoU", 0.30, 0.90, 0.70, 0.05)
        imgsz = st.selectbox("imgsz", [640, 768, 960, 1280], index=0)

    else:
        st.markdown("## Pose model")
        # Pretrained pose models ship with Ultralytics (no training needed)
        pose_model_name = st.selectbox("Model", ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"], index=0)
        pose_conf = st.slider("Pose confidence", 0.05, 0.90, 0.25, 0.05)
        pose_imgsz = st.selectbox("Pose imgsz", [640, 768, 960, 1280], index=0)
        pose_tracker = st.selectbox("Pose tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

    st.markdown("## Playback")
    adaptive = st.toggle("Adaptive real-time (auto skip)", value=True)
    max_skip = st.selectbox("Max skip", [1, 2, 3, 4, 6, 8, 12], index=3)
    base_stride = st.selectbox("Base stride", [1, 2, 3, 4], index=0)

    st.markdown("### Seek")
    st.caption("Tip: Pause → Seek → Play. Seeking while playing also works (it will jump).")

# --- State ---
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "playing" not in st.session_state:
    st.session_state.playing = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "frame_pos" not in st.session_state:
    st.session_state.frame_pos = 0
if "last_proc" not in st.session_state:
    st.session_state.last_proc = 0.0
if "seek_to" not in st.session_state:
    st.session_state.seek_to = None
if "last_frame_rgb" not in st.session_state:
    st.session_state.last_frame_rgb = None
if "last_people" not in st.session_state:
    st.session_state.last_people = 0

@st.cache_resource
def load_model(name: str):
    return YOLO(name)

# Lazy-load depending on mode
if mode == "People tracking (YOLO26)":
    model = load_model(people_model_name)
else:
    model = load_model(pose_model_name)

# --- Save upload to temp file once ---
if uploaded is not None and st.session_state.video_path is None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    st.session_state.video_path = tfile.name
    st.session_state.frame_pos = 0

# --- Controls ---
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.playing = True
        st.session_state.paused = False
with c2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.paused = not st.session_state.paused
with c3:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.playing = False
        st.session_state.paused = False

frame_box = st.empty()

# Keep last frame visible across Streamlit reruns (e.g., when pressing Pause)
if st.session_state.last_frame_rgb is not None:
    frame_box.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)

# Stats row
s1, s2, s3, s4, s5 = st.columns(5)
stat_fps = s1.metric("App FPS", "-")
stat_subjects = s2.metric("People", "-")
stat_mode = s3.metric("Mode", mode)
stat_frame = s4.metric("Frame", "-")
stat_model = s5.metric("Model", "-")

status = st.empty()

if st.session_state.video_path is None:
    st.info("Upload a video to begin.")
    st.stop()

cap = cv2.VideoCapture(st.session_state.video_path)
if not cap.isOpened():
    st.error("Could not open the video. Try MP4 (H.264).")
    st.stop()

# Video info
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

# Seek UI (needs frame_count)
if frame_count > 0:
    seek_val = st.sidebar.slider(
        "Position (frame)",
        min_value=0,
        max_value=max(0, frame_count - 1),
        value=int(min(st.session_state.frame_pos, frame_count - 1)),
        step=1,
        key="seek_slider",
    )
    if st.sidebar.button("⏩ Seek", use_container_width=True):
        st.session_state.seek_to = int(seek_val)
else:
    st.sidebar.info("Seek disabled (unknown frame count).")

# Resume from last position
cap.set(cv2.CAP_PROP_POS_FRAMES, int(st.session_state.frame_pos))

# Target wall-clock duration for a rendered step at base_stride (1x)
target_dt = (float(base_stride) / float(fps)) if fps > 0 else (1.0 / 30.0)


def draw_people(frame_bgr):
    # Track people only (COCO class 0)
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
    frame_box.image(annotated_rgb, channels="RGB", use_container_width=True)
    st.session_state.last_frame_rgb = annotated_rgb
    st.session_state.last_people = n_people
    return n_people


def draw_pose(frame_bgr):
    # Pose tracking (keypoints)
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
    frame_box.image(annotated_rgb, channels="RGB", use_container_width=True)
    st.session_state.last_frame_rgb = annotated_rgb
    st.session_state.last_people = n_people
    return n_people


# --- Playback loop ---
# Updates ONLY the frame placeholder, avoiding full-page blinking.
while True:
    if not st.session_state.playing:
        status.info("Stopped. Press Play.")
        break

    # Handle seek requests (works in stopped/paused/playing)
    if st.session_state.seek_to is not None and frame_count > 0:
        new_pos = int(max(0, min(st.session_state.seek_to, frame_count - 1)))
        st.session_state.seek_to = None
        st.session_state.frame_pos = new_pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st.session_state.frame_pos))
        ok, frame = cap.read()
        if ok and frame is not None:
            try:
                if mode == "People tracking (YOLO26)":
                    _ = draw_people(frame)
                    active_model = people_model_name
                else:
                    _ = draw_pose(frame)
                    active_model = pose_model_name
                stat_mode.metric("Mode", mode)
                stat_model.metric("Model", active_model)
                stat_frame.metric("Frame", f"{st.session_state.frame_pos}")
            except Exception as e:
                status.error(f"Seek inference error: {e}")
                break
            status.info(f"Seeked to frame {st.session_state.frame_pos}.")
        else:
            status.error("Seek failed (could not read frame).")
            break

    if st.session_state.paused:
        # Keep the last rendered frame visible while paused (even after reruns)
        if st.session_state.last_frame_rgb is not None:
            frame_box.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
        status.warning("Paused. You can Seek, then press Pause again or Play to resume.")
        time.sleep(0.1)
        continue

    # Decide how many frames to advance this tick
    eff_stride = int(base_stride)
    if adaptive:
        last_proc = float(st.session_state.last_proc or 0.0)
        ratio = last_proc / max(1e-6, target_dt) if last_proc > 0 else 1.0
        eff_stride = max(int(base_stride), min(int(max_skip), int(round(ratio))))
        eff_stride = max(1, eff_stride)

    frame = None
    for _ in range(int(eff_stride)):
        ok, f = cap.read()
        if not ok:
            st.session_state.playing = False
            frame = None
            break
        frame = f
        st.session_state.frame_pos += 1

    if frame is None:
        status.success("Finished.")
        break

    t0 = time.time()

    try:
        if mode == "People tracking (YOLO26)":
            n = draw_people(frame)
            active_model = people_model_name
        else:
            n = draw_pose(frame)
            active_model = pose_model_name
    except Exception as e:
        status.error(f"Inference error: {e}")
        break

    proc = time.time() - t0
    st.session_state.last_proc = proc

    app_fps = 1.0 / max(1e-6, proc)
    stat_fps.metric("App FPS", f"{app_fps:.1f}")
    stat_subjects.metric("People", str(n))
    stat_mode.metric("Mode", mode)
    stat_frame.metric("Frame", f"{st.session_state.frame_pos}")
    stat_model.metric("Model", active_model)

    status.info(
        f"Playing… frame {st.session_state.frame_pos} • skip {eff_stride} • proc {proc*1000:.0f} ms • target {1.0/max(1e-6,target_dt):.1f} fps"
    )

    # Real-time pacing: sleep only if we are faster than real-time
    time.sleep(max(0.0, target_dt - proc))

cap.release()
