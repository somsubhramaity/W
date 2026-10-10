"""Run Factory Safety Monitor on a recorded video and save annotated output."""

import argparse
from pathlib import Path

import cv2

from alerts import AlertManager
from pipeline import BASE_DIR, FactorySafetyPipeline
from video_summary import VideoSummaryTracker, save_summary, send_summary_telegram


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--output", default="outputs/factory_safety_annotated.mp4")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--zone", default="Factory Floor")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = BASE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        raise ValueError("Output path must not overwrite the source video")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    pipeline = FactorySafetyPipeline(config_path=args.config)
    alerts = AlertManager(config=pipeline.config)
    summary_tracker = VideoSummaryTracker(
        video_path=source,
        camera_id=args.camera_id,
        zone=args.zone,
        config=pipeline.config,
    )
    frame_count = 0
    alert_count = 0
    people_seen = 0
    people_compliant = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            annotated, detections, frame_info = pipeline.process_frame(
                frame,
                camera_id=args.camera_id,
                zone=args.zone,
            )
            writer.write(annotated)
            for person in frame_info["persons"]:
                people_seen += 1
                people_compliant += person["status"] == "compliant"
            summary_tracker.update(
                frame_index=frame_count,
                video_time_seconds=(frame_count - 1) / fps,
                persons=frame_info["persons"],
                detections=detections,
                frame=frame,
                hazard_inference_performed=frame_info["hazard_inference_performed"],
            )
            alert_count += len(
                alerts.process_frame(
                    detections,
                    annotated,
                    inference_performed=True,
                    camera_id=args.camera_id,
                    hazard_inference_performed=frame_info["hazard_inference_performed"],
                )
            )
    finally:
        capture.release()
        writer.release()

    compliance = 100 * people_compliant / people_seen if people_seen else 0
    summary = summary_tracker.build(fps)
    summary_path = save_summary(summary)
    print(f"Processed {frame_count} frames")
    print(f"Annotated video: {output}")
    print(f"Alerts logged: {alert_count}")
    print(f"Observed compliance: {compliance:.1f}% ({people_compliant}/{people_seen} person frames)")
    print(f"Alert log: {BASE_DIR / 'outputs' / 'alerts.csv'}")
    print(f"Summary JSON: {summary_path}")
    send_summary_telegram(summary)


if __name__ == "__main__":
    main()