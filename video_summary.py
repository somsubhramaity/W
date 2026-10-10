"""Track-level video statistics and end-of-video Telegram summaries."""

from datetime import datetime
import json
import os
from pathlib import Path

import cv2

from pipeline import BASE_DIR, load_config


def format_video_time(seconds):
    milliseconds = int(max(0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


class VideoSummaryTracker:
    def __init__(self, video_path, camera_id, zone, config=None, output_dir=None):
        self.config = config or load_config()
        tracking = self.config.get("tracking", {})
        self.minimum_person_frames = max(1, int(tracking.get("minimum_person_frames", 15)))
        self.missing_gear_ratio = float(tracking.get("missing_gear_ratio", 0.5))
        self.video_name = Path(video_path).name
        self.camera_id = camera_id
        self.zone = zone
        self.output_dir = Path(output_dir) if output_dir else BASE_DIR / "outputs"
        if not self.output_dir.is_absolute():
            self.output_dir = BASE_DIR / self.output_dir
        self.records = {}
        self.frame_count = 0
        self.event_counts = {"FIRE": 0, "SMOKE": 0}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def update(
        self,
        frame_index,
        video_time_seconds,
        persons,
        detections,
        frame,
        hazard_inference_performed=False,
    ):
        self.frame_count = max(self.frame_count, int(frame_index))
        if hazard_inference_performed:
            for detection in detections:
                event_type = detection.get("event_type")
                if event_type in self.event_counts:
                    self.event_counts[event_type] += 1

        frame_time = format_video_time(video_time_seconds)
        frame_height, frame_width = frame.shape[:2]
        for person in persons:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            record = self.records.get(track_id)
            if record is None:
                record = {
                    "track_id": track_id,
                    "first_seen_frame": int(frame_index),
                    "first_seen_time": frame_time,
                    "last_seen_frame": int(frame_index),
                    "last_seen_time": frame_time,
                    "frames_seen": 0,
                    "helmet_missing_frames": 0,
                    "vest_missing_frames": 0,
                    "best_snapshot_confidence": -1.0,
                    "_best_snapshot": None,
                }
                self.records[track_id] = record

            record["last_seen_frame"] = int(frame_index)
            record["last_seen_time"] = frame_time
            record["frames_seen"] += 1
            helmet_missing = bool(person.get("helmet_missing", "helmet" in person.get("missing_gear", [])))
            vest_missing = bool(person.get("vest_missing", "vest" in person.get("missing_gear", [])))
            record["helmet_missing_frames"] += helmet_missing
            record["vest_missing_frames"] += vest_missing

            if helmet_missing:
                x1, y1, x2, y2 = (int(value) for value in person["bbox"])
                x1, x2 = max(0, x1), min(frame_width, x2)
                y1, y2 = max(0, y1), min(frame_height, y2)
                confidence = float(person.get("confidence", 0.0))
                if x2 > x1 and y2 > y1 and confidence > record["best_snapshot_confidence"]:
                    record["best_snapshot_confidence"] = confidence
                    record["_best_snapshot"] = frame[y1:y2, x1:x2].copy()

    def build(self, fps):
        fps = float(fps) if fps and fps > 0 else 25.0
        duration_seconds = self.frame_count / fps
        minimum_frames = self.minimum_person_frames
        ratio_limit = self.missing_gear_ratio
        tracks = []
        without_helmet_ids = set()
        without_vest_count = 0
        snapshot_dir = self.output_dir / "summary_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for track_id, record in self.records.items():
            frames_seen = record["frames_seen"]
            helmet_ratio = record["helmet_missing_frames"] / frames_seen if frames_seen else 0.0
            vest_ratio = record["vest_missing_frames"] / frames_seen if frames_seen else 0.0
            helmet_violation = frames_seen >= minimum_frames and helmet_ratio >= ratio_limit
            vest_violation = frames_seen >= minimum_frames and vest_ratio >= ratio_limit
            if helmet_violation:
                without_helmet_ids.add(track_id)
            if vest_violation:
                without_vest_count += 1

            snapshot_path = None
            snapshot = record.pop("_best_snapshot", None)
            if snapshot is not None:
                snapshot_file = snapshot_dir / f"{self.run_id}_track_{track_id}_no_helmet.jpg"
                if cv2.imwrite(str(snapshot_file), snapshot):
                    snapshot_path = str(snapshot_file.relative_to(BASE_DIR)).replace("\\", "/")

            tracks.append(
                {
                    **record,
                    "helmet_missing_ratio": round(helmet_ratio, 4),
                    "vest_missing_ratio": round(vest_ratio, 4),
                    "without_helmet": helmet_violation,
                    "without_vest": vest_violation,
                    "helmet_status": "without helmet" if helmet_violation else "with helmet",
                    "best_snapshot_confidence": max(0.0, record["best_snapshot_confidence"]),
                    "best_snapshot": snapshot_path,
                }
            )

        top_violators = sorted(
            (track for track in tracks if track["without_helmet"]),
            key=lambda track: (
                track["helmet_missing_ratio"],
                track["helmet_missing_frames"],
                track["best_snapshot_confidence"],
            ),
            reverse=True,
        )[:3]
        total_people = len(tracks)
        hazard_count = sum(self.event_counts.values())
        return {
            "video_name": self.video_name,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "duration_seconds": round(duration_seconds, 3),
            "duration": format_video_time(duration_seconds),
            "frames_processed": self.frame_count,
            "total_unique_people": total_people,
            "without_helmet": len(without_helmet_ids),
            "with_helmet": total_people - len(without_helmet_ids),
            "without_vest": without_vest_count,
            "fire_smoke_events": {
                "detected": hazard_count > 0,
                "count": hazard_count,
                "fire": self.event_counts["FIRE"],
                "smoke": self.event_counts["SMOKE"],
            },
            "minimum_person_frames": minimum_frames,
            "missing_gear_ratio_threshold": ratio_limit,
            "tracks": sorted(tracks, key=lambda track: track["track_id"]),
            "top_violators": top_violators,
        }


def summary_message(summary):
    hazard = summary["fire_smoke_events"]
    hazard_text = (
        f"YES ({hazard['count']}; fire {hazard['fire']}, smoke {hazard['smoke']})"
        if hazard["detected"]
        else "No (0)"
    )
    return "\n".join(
        [
            "Factory Safety Monitor video summary",
            f"Video: {summary['video_name']}",
            f"Camera / zone: {summary['camera_id']} / {summary['zone']}",
            f"Duration: {summary['duration']}",
            f"Unique people tracked: {summary['total_unique_people']}",
            f"WITHOUT helmet: {summary['without_helmet']}",
            f"With helmet: {summary['with_helmet']}",
            f"Without vest: {summary['without_vest']}",
            f"Fire/smoke detected: {hazard_text}",
        ]
    )


def save_summary(summary, output_path=None):
    path = Path(output_path) if output_path else BASE_DIR / "outputs" / "summary.json"
    if not path.is_absolute():
        path = BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return path


def send_summary_telegram(summary):
    message = summary_message(summary)
    print(message)

    try:
        from dotenv import dotenv_values

        file_values = dotenv_values(BASE_DIR / ".env")
    except ImportError:
        file_values = {}
    token = os.getenv("TELEGRAM_BOT_TOKEN") or file_values.get("TELEGRAM_BOT_TOKEN")
    chat_ids = []
    for key in ("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_IDS"):
        configured_ids = os.getenv(key) or file_values.get(key)
        if configured_ids:
            chat_ids.extend(
                chat_id.strip()
                for chat_id in configured_ids.replace(";", ",").split(",")
                if chat_id.strip()
            )
    chat_ids = list(dict.fromkeys(chat_ids))
    if not token or not chat_ids:
        print("Telegram summary not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return {
            "sent": False,
            "credentials_missing": True,
            "error": "Telegram bot token or chat ID is not configured.",
        }

    try:
        import requests
    except Exception as error:
        error_type = type(error).__name__
        print(f"Telegram client setup failed ({error_type}).")
        return {
            "sent": False,
            "error": f"Telegram client unavailable ({error_type}). Check the app dependencies.",
        }

    delivery_errors = []
    successful_chats = []
    for chat_id in chat_ids:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=(5, 20),
            )
            response.raise_for_status()
            successful_chats.append(chat_id)
        except Exception as error:
            error_type = type(error).__name__
            delivery_errors.append(error_type)
            print(f"Telegram summary text failed ({error_type}).")
    if not successful_chats:
        return {
            "sent": False,
            "error": "Telegram delivery failed. Check the chat ID, bot access, and network.",
        }

    photo_errors = []
    for violator in summary.get("top_violators", []):
        snapshot_path = violator.get("best_snapshot")
        if not snapshot_path:
            continue
        image_path = BASE_DIR / snapshot_path
        if not image_path.is_file():
            continue
        for chat_id in successful_chats:
            try:
                with image_path.open("rb") as image_file:
                    response = requests.post(
                        f"https://api.telegram.org/bot{token}/sendPhoto",
                        data={
                            "chat_id": chat_id,
                            "caption": (
                                f"Track {violator['track_id']} | "
                                f"helmet missing {violator['helmet_missing_ratio']:.0%} of "
                                f"{violator['frames_seen']} frames"
                            ),
                        },
                        files={"photo": image_file},
                        timeout=(5, 30),
                    )
                    response.raise_for_status()
            except Exception as error:
                error_type = type(error).__name__
                photo_errors.append(error_type)
                print(f"Telegram violator photo failed ({error_type}).")

    return {
        "sent": True,
        "delivered_chats": len(successful_chats),
        "delivery_errors": delivery_errors,
        "photo_errors": photo_errors,
    }