import os
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import cv2


os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0",
)


def _tail_file(path: str, max_chars: int = 1200) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            data = handle.read()
        return data[-max_chars:].strip()
    except Exception:
        return ""


def _parse_rtsp_url(rtsp_url: str):
    parsed = urlparse(rtsp_url)
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 554
    return scheme, host, int(port)


def _rtsp_path_name(rtsp_url: str) -> str:
    parsed = urlparse(rtsp_url)
    return (parsed.path or "/").lstrip("/") or ""


def _hls_url_from_rtsp(rtsp_url: str, host_override: str | None = None, port: int = 8888) -> str | None:
    parsed = urlparse(rtsp_url)
    path_name = (parsed.path or "/").lstrip("/")
    if not path_name:
        return None
    host = host_override or parsed.hostname or "127.0.0.1"
    return f"http://{host}:{port}/{path_name}/index.m3u8"


def _is_local_host(host: str) -> bool:
    host_l = (host or "").lower()
    return host_l in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        if _port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def _terminate_process(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)


def _grab_frame(uri: str):
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _release_preview_capture(state):
    cap = state.get("rtsp_preview_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    state["rtsp_preview_cap"] = None
    state["rtsp_preview_url"] = None


def check_rtsp_stream_ready(rtsp_url: str, timeout_s: float = 2.0) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # Fallback when ffmpeg is unavailable.
        return _grab_frame(rtsp_url) is not None

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _local_server_has_publisher(state, rtsp_url: str) -> bool:
    log_path = state.get("rtsp_server_log_path")
    path_name = _rtsp_path_name(rtsp_url)
    if not log_path or not path_name:
        return False
    text = _tail_file(log_path, max_chars=4000)
    return f"is publishing to path '{path_name}'" in text


def stop_rtsp_server(state):
    proc = state.get("rtsp_server_process")
    _terminate_process(proc)
    state["rtsp_server_process"] = None

    if state.get("rtsp_server_kind") == "docker":
        name = state.get("rtsp_server_container_name")
        if name:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

    state["rtsp_server_kind"] = None
    state["rtsp_server_container_name"] = None


def stop_rtsp_stream(state, stop_server: bool = True):
    proc = state.get("rtsp_process")
    _terminate_process(proc)
    state["rtsp_process"] = None
    state["rtsp_streaming"] = False
    _release_preview_capture(state)
    state["rtsp_hls_url"] = None
    state["rtsp_frame_id"] = 0
    state["rtsp_fps"] = 0.0
    state["rtsp_last_frame_ts"] = None

    if stop_server:
        stop_rtsp_server(state)


def _start_local_rtsp_server_if_needed(state, rtsp_url: str):
    scheme, host, port = _parse_rtsp_url(rtsp_url)
    if scheme != "rtsp":
        return "RTSP URL must start with rtsp://"

    if not _is_local_host(host):
        return None

    if _port_open("127.0.0.1", port):
        return None

    server_log = tempfile.NamedTemporaryFile(delete=False, suffix=".rtsp-server.log")
    server_log_path = server_log.name
    server_log.close()
    state["rtsp_server_log_path"] = server_log_path

    mediamtx_bin = shutil.which("mediamtx") or shutil.which("rtsp-simple-server")
    docker_bin = shutil.which("docker")

    proc = None
    kind = None
    container_name = None

    if mediamtx_bin is not None:
        if port != 8554:
            return "Local mediamtx default port is 8554. Use rtsp://127.0.0.1:8554/... or run your own RTSP server."
        with open(server_log_path, "ab") as log_handle:
            proc = subprocess.Popen(
                [mediamtx_bin],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
            )
        kind = "mediamtx"

    elif docker_bin is not None:
        container_name = f"yolo-rtsp-{os.getpid()}"
        with open(server_log_path, "ab") as log_handle:
            proc = subprocess.Popen(
                [
                    docker_bin,
                    "run",
                    "--rm",
                    "--name",
                    container_name,
                    "-p",
                    f"{port}:8554",
                    "-p",
                    "8888:8888",
                    "bluenviron/mediamtx:latest",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
            )
        kind = "docker"

    else:
        return (
            "No RTSP server available. Install mediamtx (`brew install mediamtx`) "
            "or run your own RTSP server at this URL."
        )

    state["rtsp_server_process"] = proc
    state["rtsp_server_kind"] = kind
    state["rtsp_server_container_name"] = container_name

    if not _wait_for_port("127.0.0.1", port, timeout_s=12.0):
        stop_rtsp_server(state)
        details = _tail_file(server_log_path)
        if details:
            return f"Failed to start local RTSP server. {details}"
        return "Failed to start local RTSP server."

    return None


def start_rtsp_stream(state, video_path: str, rtsp_url: str):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return "ffmpeg was not found in PATH."

    # Current ffmpeg build publishes to an RTSP server endpoint.
    server_err = _start_local_rtsp_server_if_needed(state, rtsp_url)
    if server_err is not None:
        return server_err

    scheme, host, _ = _parse_rtsp_url(rtsp_url)
    if scheme == "rtsp":
        # Browser-native preview path for MediaMTX HLS endpoint.
        if _is_local_host(host):
            state["rtsp_hls_url"] = _hls_url_from_rtsp(rtsp_url, host_override="127.0.0.1", port=8888)
        else:
            state["rtsp_hls_url"] = _hls_url_from_rtsp(rtsp_url, host_override=host, port=8888)

    stop_rtsp_stream(state, stop_server=False)
    # stop_rtsp_stream resets hls URL, restore it.
    if scheme == "rtsp":
        if _is_local_host(host):
            state["rtsp_hls_url"] = _hls_url_from_rtsp(rtsp_url, host_override="127.0.0.1", port=8888)
        else:
            state["rtsp_hls_url"] = _hls_url_from_rtsp(rtsp_url, host_override=host, port=8888)

    stream_log = tempfile.NamedTemporaryFile(delete=False, suffix=".rtsp-stream.log")
    stream_log_path = stream_log.name
    stream_log.close()
    state["rtsp_stream_log_path"] = stream_log_path

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        rtsp_url,
    ]

    try:
        with open(stream_log_path, "ab") as log_handle:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
            )
    except Exception as exc:
        return f"Failed to start ffmpeg RTSP publisher: {exc}"

    state["rtsp_process"] = proc
    state["rtsp_streaming"] = False

    local_server_managed = state.get("rtsp_server_kind") is not None

    # Mark stream as active only after consumers can actually read frames.
    t0 = time.time()
    while (time.time() - t0) < 10.0:
        if proc.poll() is not None:
            details = _tail_file(stream_log_path)
            stop_rtsp_stream(state, stop_server=False)
            if details:
                return f"ffmpeg RTSP publisher exited early. {details}"
            return "ffmpeg RTSP publisher exited early."

        if local_server_managed:
            if _local_server_has_publisher(state, rtsp_url):
                preview = _grab_frame(rtsp_url)
                if preview is not None:
                    state["rtsp_last_preview_rgb"] = preview
                state["rtsp_streaming"] = True
                return None
        else:
            if check_rtsp_stream_ready(rtsp_url, timeout_s=1.5):
                preview = _grab_frame(rtsp_url)
                if preview is not None:
                    state["rtsp_last_preview_rgb"] = preview
                state["rtsp_streaming"] = True
                return None

        time.sleep(0.25)

    details = _tail_file(stream_log_path)
    stop_rtsp_stream(state, stop_server=False)
    if details:
        return f"RTSP stream did not become ready in time. {details}"
    return "RTSP stream did not become ready in time."


def sync_rtsp_process_state(state):
    proc = state.get("rtsp_process")
    if proc is not None and proc.poll() is not None:
        state["rtsp_process"] = None
        state["rtsp_streaming"] = False
        _release_preview_capture(state)
        details = _tail_file(state.get("rtsp_stream_log_path"))
        if details:
            state["rtsp_last_error"] = f"RTSP publisher stopped. {details}"


def get_video_info(video_path: str):
    cap_info = cv2.VideoCapture(video_path)
    if not cap_info.isOpened():
        cap_info.release()
        return None, None
    fps_value = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count_value = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_info.release()
    return fps_value, frame_count_value


def grab_rtsp_preview_frame(state, rtsp_url: str):
    scheme, host, port = _parse_rtsp_url(rtsp_url)
    if scheme != "rtsp":
        return None
    if _is_local_host(host) and not _port_open("127.0.0.1", port):
        _release_preview_capture(state)
        return None

    cap = state.get("rtsp_preview_cap")
    cap_url = state.get("rtsp_preview_url")

    if cap is None or not cap.isOpened() or cap_url != rtsp_url:
        _release_preview_capture(state)
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        state["rtsp_preview_cap"] = cap
        state["rtsp_preview_url"] = rtsp_url

    ok, frame = cap.read()
    if not ok or frame is None:
        # Reconnect once when stream hiccups.
        _release_preview_capture(state)
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        state["rtsp_preview_cap"] = cap
        state["rtsp_preview_url"] = rtsp_url
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

    return frame
