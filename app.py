"""Simple video analytics application using YOLO models."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video analytics with YOLO models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: file path or camera index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (e.g. out.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable multi-object tracking",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames in a window (press q to quit)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device: 'cpu', '0' for first GPU, etc.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter detections to these class indices (e.g. 0 2)",
    )
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    """Open a video source (file path or camera index)."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


def draw_overlay(frame: np.ndarray, fps: float, det_count: int) -> None:
    """Draw FPS and detection count on the frame (in-place)."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  Detections: {det_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


class Analytics:
    """Accumulate per-frame detection statistics."""

    def __init__(self) -> None:
        self.total_frames: int = 0
        self.detection_counts: dict[str, int] = defaultdict(int)
        self._per_frame: list[int] = []

    def update(self, labels: list[str]) -> None:
        """Record detections for one frame."""
        self.total_frames += 1
        self._per_frame.append(len(labels))
        for label in labels:
            self.detection_counts[label] += 1

    @property
    def total_detections(self) -> int:
        return sum(self._per_frame)

    @property
    def avg_detections_per_frame(self) -> float:
        if not self._per_frame:
            return 0.0
        return self.total_detections / len(self._per_frame)

    @property
    def max_detections_per_frame(self) -> int:
        return max(self._per_frame, default=0)

    def summary(self) -> str:
        """Return a human-readable analytics summary."""
        sep = "─" * 42
        lines = [
            sep,
            "  Analytics Summary",
            sep,
            f"  Total frames processed : {self.total_frames}",
            f"  Total detections       : {self.total_detections}",
            f"  Avg detections / frame : {self.avg_detections_per_frame:.2f}",
            f"  Max detections / frame : {self.max_detections_per_frame}",
        ]
        if self.detection_counts:
            lines.append("  Detections by class:")
            for cls, count in sorted(
                self.detection_counts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {cls:<22} {count}")
        lines.append(sep)
        return "\n".join(lines)


def run(args: argparse.Namespace) -> Analytics:
    """Run video analytics and return accumulated statistics."""
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    cap = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps_in, (width, height)
        )

    analytics = Analytics()
    frame_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predict_kwargs: dict = dict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                verbose=False,
            )
            if args.track:
                results = model.track(**predict_kwargs, persist=True)
            else:
                results = model.predict(**predict_kwargs)

            result = results[0]
            annotated: np.ndarray = result.plot()

            # Collect class labels for this frame
            labels: list[str] = []
            if result.boxes is not None and len(result.boxes):
                for cls_idx in result.boxes.cls.cpu().numpy().astype(int):
                    labels.append(model.names[cls_idx])
            analytics.update(labels)

            # FPS calculation
            now = time.perf_counter()
            elapsed = now - frame_time
            fps = 1.0 / elapsed if elapsed > 0 else fps
            frame_time = now

            draw_overlay(annotated, fps, len(labels))

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("YOLO Video Analytics", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(analytics.summary())
    if args.output:
        print(f"\nOutput saved to: {args.output}")

    return analytics


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
