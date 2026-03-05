import time

import cv2

from video_state import SOURCE_DIRECT, SOURCE_RTSP


def run_playback(
    state,
    source_mode: str,
    source_uri: str,
    frame_count: int,
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
        if cap is not None:
            cap.release()
