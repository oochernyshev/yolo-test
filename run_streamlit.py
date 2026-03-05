import sys
import threading
import traceback

_ORIGINAL_THREADING_EXCEPTHOOK = threading.excepthook


def _is_benign_streamlit_shutdown_exception(args):
    thread_name = args.thread.name if args.thread is not None else ""
    if thread_name != "ScriptRunner.scriptThread":
        return False

    if args.exc_type in (KeyboardInterrupt, SystemExit):
        return True

    exc_name = getattr(args.exc_type, "__name__", "")
    if exc_name in {"RerunException", "StopException"}:
        return True

    if args.exc_type is RuntimeError and str(args.exc_value) == "Event loop is closed":
        for frame in traceback.extract_tb(args.exc_traceback):
            filename = frame.filename.replace("\\", "/")
            if "site-packages/streamlit/runtime/app_session.py" in filename:
                return True
            if "site-packages/streamlit/runtime/scriptrunner/script_runner.py" in filename:
                return True

    return False


def _quiet_streamlit_threading_excepthook(args):
    # Suppress known Streamlit shutdown races; surface all other exceptions.
    if _is_benign_streamlit_shutdown_exception(args):
        return
    _ORIGINAL_THREADING_EXCEPTHOOK(args)


threading.excepthook = _quiet_streamlit_threading_excepthook

from streamlit.web import cli as stcli


def main():
    sys.argv = ["streamlit", "run", "app-video.py", *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
