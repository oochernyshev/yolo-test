import os
import tempfile
import time
from datetime import datetime

import streamlit as st

from video_inference import draw_people, draw_pose, load_model
from video_playback import run_playback
from video_rtsp import (
    grab_rtsp_preview_frame,
    get_video_info,
    start_rtsp_stream,
    stop_rtsp_stream,
    sync_rtsp_process_state,
)
from video_state import SOURCE_DIRECT, SOURCE_RTSP, init_state

st.set_page_config(page_title="Video Analytics - YOLO26 + Pose", page_icon="🎥", layout="wide")
st.title("🎥 Video Analytics: People Tracking + Pose")

init_state(st.session_state)

with st.sidebar:
    st.markdown("## Video")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])

    st.markdown("## Source")
    source_mode = st.selectbox("Input source", [SOURCE_DIRECT, SOURCE_RTSP], index=1)
    rtsp_perf_mode = False

    start_rtsp_clicked = False
    stop_rtsp_clicked = False
    if source_mode == SOURCE_RTSP:
        rtsp_url_val = st.text_input("RTSP URL", value=st.session_state.rtsp_url)
        st.session_state.rtsp_url = rtsp_url_val.strip() or st.session_state.rtsp_url
        st.caption("This uses ffmpeg to simulate a camera as an RTSP source.")
        rtsp_perf_mode = st.toggle("RTSP performance mode", value=True)
        col_stream_1, col_stream_2 = st.columns(2)
        with col_stream_1:
            start_rtsp_clicked = st.button("Start stream", width="stretch")
        with col_stream_2:
            stop_rtsp_clicked = st.button("Stop stream", width="stretch")

    st.markdown("## Mode")
    mode = st.selectbox("Analytics", ["People tracking (YOLO26)", "Pose tracking"], index=0)

    if mode == "People tracking (YOLO26)":
        st.markdown("## People model (YOLO26)")
        people_model_name = st.selectbox("Model", ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"], index=1)
        tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

        st.markdown("### Detection")
        conf = st.slider("Confidence", 0.05, 0.90, 0.15, 0.05)
        iou = st.slider("IoU", 0.30, 0.90, 0.70, 0.05)
        imgsz = st.selectbox("imgsz", [640, 768, 960, 1280], index=3)
    else:
        st.markdown("## Pose model")
        pose_model_name = st.selectbox("Model", ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"], index=0)
        pose_conf = st.slider("Pose confidence", 0.05, 0.90, 0.25, 0.05)
        pose_imgsz = st.selectbox("Pose imgsz", [640, 768, 960, 1280], index=0)
        pose_tracker = st.selectbox("Pose tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)

    st.markdown("## Playback")
    adaptive = st.toggle("Adaptive real-time (auto skip)", value=True)
    max_skip = st.selectbox("Max skip", [1, 2, 3, 4, 6, 8, 12], index=3)
    base_stride = st.selectbox("Base stride", [1, 2, 3, 4], index=0)

    st.markdown("## Export")
    export_json_enabled = st.toggle(
        "Save JSON per frame (only frames with detections)",
        value=bool(st.session_state.export_json_enabled),
    )
    export_json_dir_val = st.text_input(
        "JSON output folder",
        value=str(st.session_state.export_json_dir),
        disabled=not export_json_enabled,
    )
    minute_stats_enabled = st.toggle(
        "Save minute statistics JSON",
        value=bool(st.session_state.minute_stats_enabled),
    )
    minute_stats_dir_val = st.text_input(
        "Minute stats output folder",
        value=str(st.session_state.minute_stats_dir),
        disabled=not minute_stats_enabled,
    )

    st.markdown("### Seek")
    st.caption("Tip: Pause -> Seek -> Start processing. Seek is only available in Direct file mode.")

st.session_state.export_json_enabled = bool(export_json_enabled)
if str(export_json_dir_val).strip():
    st.session_state.export_json_dir = str(export_json_dir_val).strip()
st.session_state.minute_stats_enabled = bool(minute_stats_enabled)
if str(minute_stats_dir_val).strip():
    st.session_state.minute_stats_dir = str(minute_stats_dir_val).strip()


def _prepare_json_export_run(state, mode_value):
    if not state["export_json_enabled"]:
        state["export_json_session_dir"] = None
        state["export_json_written"] = 0
        state["export_json_last_error"] = None
        return

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "rtsp" if mode_value == SOURCE_RTSP else "direct"
    session_dir = os.path.join(str(state["export_json_dir"]), f"{mode_tag}_{run_stamp}")
    os.makedirs(session_dir, exist_ok=True)
    state["export_json_session_dir"] = session_dir
    state["export_json_written"] = 0
    state["export_json_last_error"] = None


def _prepare_minute_stats_run(state, mode_value):
    if not state["minute_stats_enabled"]:
        state["minute_stats_session_dir"] = None
        state["minute_stats_written"] = 0
        state["minute_stats_last_error"] = None
        state["minute_stats_current_index"] = None
        state["minute_stats_objects_total"] = 0
        state["minute_stats_processed_frames"] = 0
        state["minute_stats_frames_with_detections"] = 0
        state["minute_stats_unique_track_ids"] = []
        return

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "rtsp" if mode_value == SOURCE_RTSP else "direct"
    session_dir = os.path.join(str(state["minute_stats_dir"]), f"{mode_tag}_{run_stamp}")
    os.makedirs(session_dir, exist_ok=True)
    state["minute_stats_session_dir"] = session_dir
    state["minute_stats_written"] = 0
    state["minute_stats_last_error"] = None
    state["minute_stats_current_index"] = None
    state["minute_stats_objects_total"] = 0
    state["minute_stats_processed_frames"] = 0
    state["minute_stats_frames_with_detections"] = 0
    state["minute_stats_unique_track_ids"] = []

if st.session_state.source_mode_prev != source_mode:
    if st.session_state.source_mode_prev == SOURCE_RTSP:
        stop_rtsp_stream(st.session_state)
    st.session_state.source_mode_prev = source_mode
    st.session_state.playing = False
    st.session_state.paused = False
    st.session_state.frame_pos = 0
    st.session_state.rtsp_last_error = None
    st.session_state.rtsp_should_be_running = False

sync_rtsp_process_state(st.session_state)

if uploaded is not None:
    uploaded_signature = (uploaded.name, uploaded.size)
    if st.session_state.uploaded_signature != uploaded_signature:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        st.session_state.video_path = tfile.name
        st.session_state.uploaded_signature = uploaded_signature
        st.session_state.frame_pos = 0
        st.session_state.last_frame_rgb = None
        st.session_state.playing = False
        st.session_state.paused = False
        st.session_state.rtsp_last_error = None
        st.session_state.rtsp_should_be_running = False
        if st.session_state.rtsp_streaming:
            stop_rtsp_stream(st.session_state)

if start_rtsp_clicked:
    if st.session_state.video_path is None:
        st.sidebar.error("Upload a video first.")
    else:
        st.session_state.rtsp_last_error = None
        st.session_state.rtsp_should_be_running = True
        err = start_rtsp_stream(st.session_state, st.session_state.video_path, st.session_state.rtsp_url)
        if err is not None:
            st.session_state.rtsp_last_error = err
            st.session_state.rtsp_next_retry_ts = time.time() + 2.0
            st.sidebar.error(err)
        else:
            st.session_state.rtsp_next_retry_ts = 0.0
            st.sidebar.success("RTSP stream started.")

if stop_rtsp_clicked:
    stop_rtsp_stream(st.session_state)
    st.session_state.rtsp_last_error = None
    st.session_state.rtsp_should_be_running = False
    st.session_state.rtsp_next_retry_ts = 0.0
    st.sidebar.info("RTSP stream stopped.")

# Keep RTSP stream alive in RTSP mode: restart publisher if it exits unexpectedly.
if (
    source_mode == SOURCE_RTSP
    and st.session_state.rtsp_should_be_running
    and st.session_state.video_path is not None
    and not st.session_state.rtsp_streaming
    and time.time() >= float(st.session_state.rtsp_next_retry_ts or 0.0)
):
    err = start_rtsp_stream(st.session_state, st.session_state.video_path, st.session_state.rtsp_url)
    if err is not None:
        st.session_state.rtsp_last_error = err
        st.session_state.rtsp_next_retry_ts = time.time() + 2.0
    else:
        st.session_state.rtsp_next_retry_ts = 0.0

if mode == "People tracking (YOLO26)":
    model = load_model(people_model_name)

    def draw_frame(frame_bgr):
        eff_imgsz = min(int(imgsz), 640) if source_mode == SOURCE_RTSP and rtsp_perf_mode else int(imgsz)
        return draw_people(model, frame_bgr, frame_box, conf, iou, eff_imgsz, tracker, st.session_state)

    active_model_name = people_model_name
else:
    model = load_model(pose_model_name)

    def draw_frame(frame_bgr):
        eff_pose_imgsz = (
            min(int(pose_imgsz), 640) if source_mode == SOURCE_RTSP and rtsp_perf_mode else int(pose_imgsz)
        )
        return draw_pose(model, frame_bgr, frame_box, pose_conf, eff_pose_imgsz, pose_tracker, st.session_state)

    active_model_name = pose_model_name

if source_mode == SOURCE_RTSP:
    pcol, rcol = st.columns([1, 3])
    with pcol:
        st.markdown("#### RTSP Stream (preview)")
        m1, m2 = st.columns(2)
        rtsp_fps_metric = m1.empty()
        rtsp_frame_metric = m2.empty()
        use_browser_preview = bool(st.session_state.rtsp_hls_url)
        if use_browser_preview:
            st.caption("Browser-native HLS preview")
            st.video(st.session_state.rtsp_hls_url)
            rtsp_preview_box = None
        else:
            rtsp_preview_box = st.empty()
    with rcol:
        st.markdown("#### Processing result")
        frame_box = st.empty()
else:
    use_browser_preview = False
    rtsp_fps_metric = None
    rtsp_frame_metric = None
    rtsp_preview_box = None
    frame_box = st.empty()

if st.session_state.last_frame_rgb is not None:
    frame_box.image(st.session_state.last_frame_rgb, channels="RGB", width="stretch")

if source_mode == SOURCE_RTSP and st.session_state.rtsp_last_error:
    st.error(st.session_state.rtsp_last_error)

if st.session_state.export_json_enabled:
    if st.session_state.export_json_session_dir:
        st.caption(
            f"JSON export: {st.session_state.export_json_written} files -> "
            f"{st.session_state.export_json_session_dir}"
        )
    if st.session_state.export_json_last_error:
        st.warning(f"JSON export error: {st.session_state.export_json_last_error}")

if st.session_state.minute_stats_enabled:
    if st.session_state.minute_stats_session_dir:
        st.caption(
            f"Minute stats export: {st.session_state.minute_stats_written} files -> "
            f"{st.session_state.minute_stats_session_dir}"
        )
    if st.session_state.minute_stats_last_error:
        st.warning(f"Minute stats error: {st.session_state.minute_stats_last_error}")

can_start_processing = st.session_state.video_path is not None

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶ Start processing", width="stretch", disabled=not can_start_processing):
        st.session_state.rtsp_last_error = None
        _prepare_json_export_run(st.session_state, source_mode)
        _prepare_minute_stats_run(st.session_state, source_mode)
        if source_mode == SOURCE_RTSP and not st.session_state.rtsp_streaming:
            st.session_state.rtsp_should_be_running = True
            err = start_rtsp_stream(st.session_state, st.session_state.video_path, st.session_state.rtsp_url)
            if err is not None:
                st.session_state.rtsp_last_error = err
                st.session_state.rtsp_next_retry_ts = time.time() + 2.0
                st.session_state.playing = False
                st.session_state.paused = False
            else:
                st.session_state.rtsp_next_retry_ts = 0.0
                st.session_state.playing = True
                st.session_state.paused = False
        else:
            st.session_state.playing = True
            st.session_state.paused = False
with c2:
    if st.button("⏸ Pause", width="stretch", disabled=not st.session_state.playing):
        st.session_state.paused = not st.session_state.paused
with c3:
    if st.button("⏹ Stop", width="stretch", disabled=not st.session_state.playing):
        st.session_state.playing = False
        st.session_state.paused = False

s1, s2, s3, s4, s5 = st.columns(5)
stat_fps = s1.metric("App FPS", "-")
stat_subjects = s2.metric("People", "-")
stat_mode = s3.metric("Mode", mode)
stat_frame = s4.metric("Frame", "-")
stat_model = s5.metric("Model", "-")

status = st.empty()

if st.session_state.video_path is None:
    status.info("Upload a video to begin.")
    st.stop()

if source_mode == SOURCE_RTSP and not st.session_state.rtsp_streaming:
    status.info("Start RTSP stream first, then press Start processing.")

fps = 30.0
frame_count = 0
if source_mode == SOURCE_DIRECT:
    fps_info, frame_count_info = get_video_info(st.session_state.video_path)
    if fps_info is None:
        st.error("Could not open the video. Try MP4 (H.264).")
        st.stop()
    fps = fps_info
    frame_count = frame_count_info

    if frame_count > 0:
        seek_val = st.sidebar.slider(
            "Position (frame)",
            min_value=0,
            max_value=max(0, frame_count - 1),
            value=int(min(st.session_state.frame_pos, frame_count - 1)),
            step=1,
            key="seek_slider",
        )
        if st.sidebar.button("⏩ Seek", width="stretch"):
            st.session_state.seek_to = int(seek_val)
    else:
        st.sidebar.info("Seek disabled (unknown frame count).")
else:
    st.sidebar.info("Seek disabled for RTSP source.")

target_dt = (float(base_stride) / float(fps)) if fps > 0 else (1.0 / 30.0)
source_uri = st.session_state.video_path if source_mode == SOURCE_DIRECT else st.session_state.rtsp_url


def _update_rtsp_metrics(on_frame: bool):
    if on_frame:
        now = time.time()
        prev_ts = st.session_state.rtsp_last_frame_ts
        if prev_ts is not None:
            inst_fps = 1.0 / max(1e-6, now - prev_ts)
            old_fps = float(st.session_state.rtsp_fps or 0.0)
            st.session_state.rtsp_fps = inst_fps if old_fps <= 0 else (0.8 * old_fps + 0.2 * inst_fps)
        st.session_state.rtsp_last_frame_ts = now
        st.session_state.rtsp_frame_id = int(st.session_state.rtsp_frame_id) + 1

    if rtsp_fps_metric is not None:
        if st.session_state.rtsp_fps > 0:
            rtsp_fps_metric.metric("RTSP FPS", f"{st.session_state.rtsp_fps:.1f}")
        else:
            rtsp_fps_metric.metric("RTSP FPS", "-")
    if rtsp_frame_metric is not None:
        frame_id = int(st.session_state.rtsp_frame_id)
        rtsp_frame_metric.metric("RTSP Frame", str(frame_id) if frame_id > 0 else "-")


_update_rtsp_metrics(on_frame=False)


def _render_rtsp_preview(preview_box):
    preview_bgr = grab_rtsp_preview_frame(st.session_state, st.session_state.rtsp_url)
    if preview_bgr is not None:
        _update_rtsp_metrics(on_frame=True)
        if preview_box is not None:
            preview_box.image(preview_bgr, channels="BGR", width=320)
    elif preview_box is not None:
        preview_box.info("Waiting for RTSP preview...")

run_playback(
    state=st.session_state,
    source_mode=source_mode,
    source_uri=source_uri,
    frame_count=frame_count,
    source_fps=float(fps),
    target_dt=target_dt,
    base_stride=int(base_stride),
    max_skip=int(max_skip),
    adaptive=adaptive,
    mode=mode,
    active_model_name=active_model_name,
    draw_frame=draw_frame,
    frame_box=frame_box,
    status=status,
    stat_fps=stat_fps,
    stat_subjects=stat_subjects,
    stat_mode=stat_mode,
    stat_frame=stat_frame,
    stat_model=stat_model,
    rtsp_preview_box=None if use_browser_preview else rtsp_preview_box,
    on_rtsp_frame=lambda: _update_rtsp_metrics(on_frame=True),
)


if source_mode == SOURCE_RTSP and st.session_state.rtsp_streaming and not st.session_state.playing and not use_browser_preview:
    if hasattr(st, "fragment"):
        @st.fragment(run_every=0.10)
        def _rtsp_preview_fragment():
            _render_rtsp_preview(rtsp_preview_box)

        _rtsp_preview_fragment()
    else:
        # Fallback for older Streamlit: full rerun refresh.
        _render_rtsp_preview(rtsp_preview_box)
        time.sleep(0.2)
        st.rerun()
