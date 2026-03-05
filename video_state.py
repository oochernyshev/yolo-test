SOURCE_DIRECT = "Direct file"
SOURCE_RTSP = "RTSP simulated camera"


def init_state(state):
    defaults = {
        "video_path": None,
        "uploaded_signature": None,
        "playing": False,
        "paused": False,
        "frame_pos": 0,
        "last_proc": 0.0,
        "seek_to": None,
        "last_frame_rgb": None,
        "last_people": 0,
        "rtsp_url": "rtsp://127.0.0.1:8554/simcam",
        "rtsp_process": None,
        "rtsp_streaming": False,
        "rtsp_last_preview_rgb": None,
        "rtsp_preview_cap": None,
        "rtsp_preview_url": None,
        "rtsp_hls_url": None,
        "rtsp_should_be_running": False,
        "rtsp_next_retry_ts": 0.0,
        "rtsp_frame_id": 0,
        "rtsp_fps": 0.0,
        "rtsp_last_frame_ts": None,
        "rtsp_last_error": None,
        "rtsp_stream_log_path": None,
        "rtsp_server_process": None,
        "rtsp_server_kind": None,
        "rtsp_server_container_name": None,
        "rtsp_server_log_path": None,
        "source_mode_prev": SOURCE_RTSP,
        "last_detections": [],
        "export_json_enabled": True,
        "export_json_dir": "detections_json",
        "export_json_session_dir": None,
        "export_json_written": 0,
        "export_json_last_error": None,
        "minute_stats_enabled": True,
        "minute_stats_dir": "minute_stats_json",
        "minute_stats_session_dir": None,
        "minute_stats_written": 0,
        "minute_stats_last_error": None,
        "minute_stats_current_index": None,
        "minute_stats_objects_total": 0,
        "minute_stats_processed_frames": 0,
        "minute_stats_frames_with_detections": 0,
        "minute_stats_unique_track_ids": [],
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
