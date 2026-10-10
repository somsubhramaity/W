"""Alert debounce, snapshots, logs, and optional Telegram delivery."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time

import cv2

from pipeline import BASE_DIR, load_config


SEVERITY = {
    "FIRE": "critical",
    "SMOKE": "high",
    "NO_HELMET": "medium",
    "NO_VEST": "medium",
    "NO_BOOTS": "medium",
    "NO_GLOVES": "medium",
}
LOG_FIELDS = [
    "event_type",
    "severity",
    "camera_id",
    "zone",
    "timestamp",
    "confidence",
    "snapshot",
]


class AlertManager:
    def __init__(self, config=None, output_dir=None, clock=None):
        self.config = config or load_config()
        self.output_dir = Path(output_dir) if output_dir else BASE_DIR / "outputs"
        if not self.output_dir.is_absolute():
            self.output_dir = BASE_DIR / self.output_dir
        self.snapshot_dir = self.output_dir / "alerts"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        alert_config = self.config.get("alerts", {})
        self.cooldown_seconds = float(alert_config.get("cooldown_seconds", 10))
        self.confirmation_frames = max(1, int(alert_config.get("confirmation_frames", 3)))
        self.clock = clock or time.time
        self._last_alert_time = {}
        self._confirmation_counts = {}

    def process_frame(
        self,
        detections,
        frame,
        inference_performed=True,
        camera_id=None,
        hazard_inference_performed=True,
    ):
        if not inference_performed:
            return []

        candidates = {}
        camera_ids = {str(camera_id)} if camera_id is not None else set()
        for detection in detections:
            event_type = detection.get("event_type")
            if event_type not in SEVERITY:
                continue
            camera_id = str(detection.get("camera_id", "camera-1"))
            camera_ids.add(camera_id)
            key = (camera_id, event_type)
            previous = candidates.get(key)
            if previous is None or detection.get("confidence", 0) > previous.get("confidence", 0):
                candidates[key] = detection

        if hazard_inference_performed:
            for key in list(self._confirmation_counts):
                key_camera_id, event_type = key
                if key_camera_id in camera_ids and event_type in ("FIRE", "SMOKE") and key not in candidates:
                    self._confirmation_counts[key] = 0

        alerts = []
        now = self.clock()
        for (camera_id, event_type), detection in candidates.items():
            if event_type in ("FIRE", "SMOKE"):
                if not hazard_inference_performed:
                    continue
                key = (camera_id, event_type)
                self._confirmation_counts[key] = self._confirmation_counts.get(key, 0) + 1
                if self._confirmation_counts[key] < self.confirmation_frames:
                    continue
            key = (camera_id, event_type)
            if now - self._last_alert_time.get(key, float("-inf")) < self.cooldown_seconds:
                continue
            alert = self._create_alert(detection, frame)
            self._last_alert_time[key] = now
            alerts.append(alert)
            self._write_logs(alert)
            self._notify_telegram(alert)
            print(
                f"[{alert['severity'].upper()}] {event_type} | "
                f"{alert['camera_id']} / {alert['zone']} | {alert['timestamp']}"
            )
        return alerts

    def _create_alert(self, detection, frame):
        event_type = detection["event_type"]
        camera_id = str(detection.get("camera_id", "camera-1"))
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        safe_camera = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_camera}_{event_type.lower()}.jpg"
        snapshot_path = self.snapshot_dir / filename
        if not cv2.imwrite(str(snapshot_path), frame):
            raise OSError(f"Could not save alert snapshot: {snapshot_path}")
        return {
            "event_type": event_type,
            "severity": SEVERITY[event_type],
            "camera_id": camera_id,
            "zone": detection.get("zone", "Unassigned"),
            "timestamp": timestamp,
            "confidence": float(detection.get("confidence", 0.0)),
            "snapshot": str(snapshot_path.relative_to(BASE_DIR)).replace("\\", "/"),
        }

    def _write_logs(self, alert):
        csv_path = self.output_dir / "alerts.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(alert)

        json_path = self.output_dir / "alerts.json"
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as json_file:
                    entries = json.load(json_file)
            except (json.JSONDecodeError, OSError):
                entries = []
        else:
            entries = []
        entries.append(alert)
        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(entries, json_file, indent=2)

    def _notify_telegram(self, alert):
        if not self.config.get("alerts", {}).get("telegram_live_enabled", False):
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(BASE_DIR / ".env")
        except ImportError:
            return
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        try:
            import requests

            caption = (
                f"{alert['severity'].upper()} {alert['event_type']} | "
                f"{alert['camera_id']} / {alert['zone']} | {alert['timestamp']}"
            )
            snapshot_path = BASE_DIR / alert["snapshot"]
            with snapshot_path.open("rb") as snapshot:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": snapshot},
                    timeout=10,
                ).raise_for_status()
        except Exception as error:
            print(f"Telegram notification failed: {error}")