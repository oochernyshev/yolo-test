import json
import os
import time
from datetime import datetime, timezone

import cv2

from video_state import SOURCE_DIRECT, SOURCE_RTSP


def _write_detection_json(
    state,
    mode: str,
    source_mode: str,
    source_uri: str,
    frame_pos: int,
    active_model_name: str,
    detections,
):
    session_dir = state.get("export_json_session_dir")
    if not session_dir:
        return "JSON export directory is not set."

    try:
        os.makedirs(session_dir, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame": int(frame_pos),
            "mode": mode,
            "model": active_model_name,
            "source_mode": source_mode,
            "source_uri": source_uri,
            "detections_count": len(detections),
            "detections": detections,
        }
        file_name = f"frame_{int(frame_pos):08d}.json"
        out_path = os.path.join(session_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        state["export_json_written"] = int(state.get("export_json_written", 0)) + 1
        state["export_json_last_error"] = None
    except Exception as exc:
        return str(exc)

    return None


def _reset_minute_stats_bucket(state, minute_index: int):
    state["minute_stats_current_index"] = int(minute_index)
    state["minute_stats_objects_total"] = 0
    state["minute_stats_processed_frames"] = 0
    state["minute_stats_frames_with_detections"] = 0
    state["minute_stats_unique_track_ids"] = []


def _write_minute_stats_json(
    state,
    mode: str,
    source_mode: str,
    source_uri: str,
    active_model_name: str,
    source_fps: float,
    minute_index: int,
    is_partial: bool,
):
    session_dir = state.get("minute_stats_session_dir")
    if not session_dir:
        return "Minute stats export directory is not set."

    try:
        os.makedirs(session_dir, exist_ok=True)
        track_ids = sorted(
            {
                int(v)
                for v in (state.get("minute_stats_unique_track_ids") or [])
                if isinstance(v, int) or (isinstance(v, str) and v.lstrip("-").isdigit())
            }
        )
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "minute_index": int(minute_index),
            "minute_start_sec": int(minute_index) * 60,
            "minute_end_sec": (int(minute_index) + 1) * 60,
            "is_partial": bool(is_partial),
            "mode": mode,
            "model": active_model_name,
            "source_mode": source_mode,
            "source_uri": source_uri,
            "source_fps": float(source_fps),
            "objects_detected_total": int(state.get("minute_stats_objects_total", 0)),
            "processed_frames": int(state.get("minute_stats_processed_frames", 0)),
            "frames_with_detections": int(state.get("minute_stats_frames_with_detections", 0)),
            "unique_track_ids_count": len(track_ids),
            "unique_track_ids": track_ids,
        }
        file_name = f"minute_{int(minute_index):04d}.json"
        out_path = os.path.join(session_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        state["minute_stats_written"] = int(state.get("minute_stats_written", 0)) + 1
        state["minute_stats_last_error"] = None
    except Exception as exc:
        return str(exc)

    return None


def _update_minute_stats(
    state,
    mode: str,
    source_mode: str,
    source_uri: str,
    active_model_name: str,
    source_fps: float,
    frame_pos: int,
    detections,
):
    if not state.get("minute_stats_enabled"):
        return None

    fps_value = float(source_fps) if source_fps and float(source_fps) > 0 else 30.0
    frames_per_minute = max(1, int(round(fps_value * 60.0)))
    minute_index = int(max(0, int(frame_pos) - 1) // frames_per_minute)

    current_index = state.get("minute_stats_current_index")
    if current_index is None:
        _reset_minute_stats_bucket(state, minute_index)
    elif int(current_index) != minute_index:
        flush_err = _write_minute_stats_json(
            state=state,
            mode=mode,
            source_mode=source_mode,
            source_uri=source_uri,
            active_model_name=active_model_name,
            source_fps=fps_value,
            minute_index=int(current_index),
            is_partial=False,
        )
        _reset_minute_stats_bucket(state, minute_index)
        if flush_err is not None:
            return flush_err

    state["minute_stats_processed_frames"] = int(state.get("minute_stats_processed_frames", 0)) + 1

    detected_count = len(detections)
    state["minute_stats_objects_total"] = int(state.get("minute_stats_objects_total", 0)) + int(detected_count)
    if detected_count > 0:
        state["minute_stats_frames_with_detections"] = int(state.get("minute_stats_frames_with_detections", 0)) + 1

    existing_ids = set()
    for value in state.get("minute_stats_unique_track_ids") or []:
        try:
            existing_ids.add(int(value))
        except Exception:
            continue
    for det in detections:
        track_id = det.get("track_id") if isinstance(det, dict) else None
        if track_id is None:
            continue
        try:
            existing_ids.add(int(track_id))
        except Exception:
            continue
    state["minute_stats_unique_track_ids"] = sorted(existing_ids)

    return None


def run_playback(
    state,
    source_mode: str,
    source_uri: str,
    frame_count: int,
    source_fps: float,
    target_dt: float,
    base_stride: int,
    max_skip: int,
    adaptive: bool,
    mode: str,
    active_model_name: str,
    draw_frame,
    frame_box,
    status,
    stat_fps,
    stat_subjects,
    stat_mode,
    stat_frame,
    stat_model,
    rtsp_preview_box=None,
    on_rtsp_frame=None,
):
    cap = None
    try:
        if state.playing:
            if source_mode == SOURCE_RTSP:
                preview_cap = state.get("rtsp_preview_cap")
                if preview_cap is not None:
                    try:
                        preview_cap.release()
                    except Exception:
                        pass
                    state["rtsp_preview_cap"] = None
                    state["rtsp_preview_url"] = None

            attempts = 1 if source_mode == SOURCE_DIRECT else 10
            for _ in range(attempts):
                cap = cv2.VideoCapture(source_uri)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
                time.sleep(0.2)

            if cap is None or not cap.isOpened():
                state.playing = False
                status.error("Could not open processing source.")
            elif source_mode == SOURCE_DIRECT:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(state.frame_pos))

        while True:
            if not state.playing:
                if source_mode == SOURCE_RTSP and state.rtsp_streaming:
                    status.info("RTSP stream is running. Press Start processing.")
                else:
                    status.info("Stopped. Press Start processing.")
                break

            if cap is None:
                state.playing = False
                status.error("Source capture is not ready.")
                break

            if source_mode == SOURCE_DIRECT and state.seek_to is not None and frame_count > 0:
                new_pos = int(max(0, min(state.seek_to, frame_count - 1)))
                state.seek_to = None
                state.frame_pos = new_pos
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(state.frame_pos))
                ok, seek_frame = cap.read()
                if ok and seek_frame is not None:
                    try:
                        _ = draw_frame(seek_frame)
                    except Exception as exc:
                        status.error(f"Seek inference error: {exc}")
                        state.playing = False
                        break
                    status.info(f"Seeked to frame {state.frame_pos}.")
                else:
                    status.error("Seek failed (could not read frame).")
                    state.playing = False
                    break

            if state.paused:
                if state.last_frame_rgb is not None:
                    frame_box.image(state.last_frame_rgb, channels="RGB", width="stretch")
                status.warning("Paused. Press Pause again to resume.")
                time.sleep(0.1)
                continue

            eff_stride = int(base_stride)
            if adaptive:
                last_proc = float(state.last_proc or 0.0)
                ratio = last_proc / max(1e-6, target_dt) if last_proc > 0 else 1.0
                eff_stride = max(int(base_stride), min(int(max_skip), int(round(ratio))))
                eff_stride = max(1, eff_stride)

            frame = None
            if source_mode == SOURCE_DIRECT:
                for _ in range(int(eff_stride)):
                    ok, frame_candidate = cap.read()
                    if not ok:
                        frame = None
                        break
                    frame = frame_candidate
                    state.frame_pos += 1

                if frame is None:
                    status.success("Finished.")
                    state.playing = False
                    break
            else:
                for _ in range(int(eff_stride)):
                    ok, frame_candidate = cap.read()
                    if not ok:
                        frame = None
                        break
                    frame = frame_candidate
                    state.frame_pos += 1

                if frame is None:
                    status.warning("Waiting for RTSP frames...")
                    cap.release()
                    time.sleep(0.1)
                    cap = cv2.VideoCapture(source_uri)
                    continue

            if source_mode == SOURCE_RTSP and rtsp_preview_box is not None and frame is not None:
                rtsp_preview_box.image(frame, channels="BGR", width=320)
            if source_mode == SOURCE_RTSP and frame is not None and on_rtsp_frame is not None:
                on_rtsp_frame()

            t0 = time.time()
            try:
                n_people = draw_frame(frame)
            except Exception as exc:
                status.error(f"Inference error: {exc}")
                state.playing = False
                break

            detections = state.get("last_detections") or []
            if state.get("export_json_enabled") and detections:
                export_err = _write_detection_json(
                    state=state,
                    mode=mode,
                    source_mode=source_mode,
                    source_uri=source_uri,
                    frame_pos=int(state.frame_pos),
                    active_model_name=active_model_name,
                    detections=detections,
                )
                if export_err and export_err != state.get("export_json_last_error"):
                    state["export_json_last_error"] = export_err
                    status.warning(f"JSON export error: {export_err}")

            minute_err = _update_minute_stats(
                state=state,
                mode=mode,
                source_mode=source_mode,
                source_uri=source_uri,
                active_model_name=active_model_name,
                source_fps=source_fps,
                frame_pos=int(state.frame_pos),
                detections=detections,
            )
            if minute_err and minute_err != state.get("minute_stats_last_error"):
                state["minute_stats_last_error"] = minute_err
                status.warning(f"Minute stats error: {minute_err}")

            proc = time.time() - t0
            state.last_proc = proc

            app_fps = 1.0 / max(1e-6, proc)
            stat_fps.metric("App FPS", f"{app_fps:.1f}")
            stat_subjects.metric("People", str(n_people))
            stat_mode.metric("Mode", mode)
            stat_frame.metric("Frame", f"{state.frame_pos}")
            stat_model.metric("Model", active_model_name)

            status.info(
                f"Processing... frame {state.frame_pos} • skip {eff_stride} • "
                f"proc {proc * 1000:.0f} ms • target {1.0 / max(1e-6, target_dt):.1f} fps"
            )

            time.sleep(max(0.0, target_dt - proc))
    except KeyboardInterrupt:
        state.playing = False
        state.paused = False
    finally:
        if (
            state.get("minute_stats_enabled")
            and state.get("minute_stats_current_index") is not None
            and int(state.get("minute_stats_processed_frames", 0)) > 0
        ):
            final_err = _write_minute_stats_json(
                state=state,
                mode=mode,
                source_mode=source_mode,
                source_uri=source_uri,
                active_model_name=active_model_name,
                source_fps=source_fps,
                minute_index=int(state.get("minute_stats_current_index")),
                is_partial=True,
            )
            if final_err and final_err != state.get("minute_stats_last_error"):
                state["minute_stats_last_error"] = final_err
            if final_err is None:
                state["minute_stats_current_index"] = None
                state["minute_stats_objects_total"] = 0
                state["minute_stats_processed_frames"] = 0
                state["minute_stats_frames_with_detections"] = 0
                state["minute_stats_unique_track_ids"] = []
        if cap is not None:
            cap.release()
