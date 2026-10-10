"""Combined PPE and fire/smoke detection for video frames."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

import cv2
import yaml

from fire_smoke import FireSmokeDetector


BASE_DIR = Path(__file__).resolve().parent
GEAR_CLASSES = {"helmet", "vest", "boots", "gloves"}
NEGATIVE_CLASSES = {
    "no_helmet": "helmet",
    "no_boots": "boots",
    "no_gloves": "gloves",
}
EVENT_TYPES = {
    "no_helmet": "NO_HELMET",
    "no_boots": "NO_BOOTS",
    "no_gloves": "NO_GLOVES",
    "no_vest": "NO_VEST",
}


def load_config(config_path=None):
    path = Path(config_path) if config_path else BASE_DIR / "config.yaml"
    if not path.is_absolute():
        path = BASE_DIR / path
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    config["_base_dir"] = path.parent
    return config


def _box_values(box):
    return tuple(int(value) for value in box)


def _contains(outer_box, inner_box):
    center_x = (inner_box[0] + inner_box[2]) / 2
    center_y = (inner_box[1] + inner_box[3]) / 2
    return outer_box[0] <= center_x <= outer_box[2] and outer_box[1] <= center_y <= outer_box[3]


class FactorySafetyPipeline:
    def __init__(self, config_path=None, ppe_model=None, fire_smoke_detector=None):
        self.config = load_config(config_path)
        base_dir = self.config["_base_dir"]
        models = self.config.get("models", {})
        inference = self.config.get("inference", {})
        self.confidence = float(inference.get("confidence", 0.25))
        self.image_size = min(640, int(inference.get("image_size", 640)))
        self.frame_stride = max(1, int(inference.get("frame_stride", 1)))
        self.required_gear = set(self.config.get("required_gear", ["helmet", "vest"]))
        self.ppe_model = ppe_model or self._load_ppe_model(base_dir, models.get("ppe"))
        self.fire_smoke_detector = fire_smoke_detector or FireSmokeDetector(
            model_path=self._model_path(base_dir, models.get("fire_smoke")),
            confidence=self.confidence,
            image_size=self.image_size,
        )
        self.frame_count = 0
        self.last_hazards = []
        self.last_persons = []
        self.last_fps = 0.0

    @staticmethod
    def _model_path(base_dir, configured_path):
        path = Path(configured_path or "models/ppe_yolov8.pt")
        return path if path.is_absolute() else base_dir / path

    def _load_ppe_model(self, base_dir, configured_path):
        model_path = self._model_path(base_dir, configured_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"PPE model not found: {model_path}. Restore or train models/ppe_yolov8.pt first."
            )
        from ultralytics import YOLO

        return YOLO(str(model_path))

    @staticmethod
    def _model_detections(model, frame, confidence, image_size):
        detections = []
        # Track all classes in one pass to retain PPE boxes and person IDs together.
        results = model.track(
            source=frame,
            conf=confidence,
            imgsz=image_size,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                detection = {
                    "class": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": _box_values(box.xyxy[0]),
                }
                if class_name.lower() == "person" and box.id is not None:
                    detection["track_id"] = int(box.id[0])
                detections.append(detection)
        return detections

    def _enrich_detections(self, raw_detections, camera_id, zone, timestamp):
        persons = [item for item in raw_detections if item["class"].lower() == "person"]
        gear = [item for item in raw_detections if item["class"].lower() in GEAR_CLASSES]
        negatives = [item for item in raw_detections if item["class"].lower() in NEGATIVE_CLASSES]
        enriched = []
        person_summaries = []

        for index, person in enumerate(persons, start=1):
            person_box = person["bbox"]
            present = {
                item["class"].lower()
                for item in gear
                if _contains(person_box, item["bbox"])
            }
            explicit_missing = {
                NEGATIVE_CLASSES[item["class"].lower()]
                for item in negatives
                if _contains(person_box, item["bbox"])
            }
            missing = explicit_missing | (self.required_gear - present)
            status = "violation" if missing else "compliant"
            summary = {
                "person_id": person.get("track_id", index),
                "track_id": person.get("track_id"),
                "bbox": person_box,
                "confidence": person["confidence"],
                "status": status,
                "missing_gear": sorted(missing),
                "present_gear": sorted(present),
                "helmet_missing": "helmet" not in present or "helmet" in explicit_missing,
                "vest_missing": "vest" not in present or "vest" in explicit_missing,
            }
            person_summaries.append(summary)
            enriched.append(
                {
                    "event_type": "PERSON",
                    "class": "Person",
                    "confidence": person["confidence"],
                    "bbox": person_box,
                    "track_id": person.get("track_id"),
                    "camera_id": camera_id,
                    "zone": zone,
                    "timestamp": timestamp,
                    "compliance": status,
                    "missing_gear": sorted(missing),
                }
            )
            for missing_item in sorted(missing):
                event_class = f"no_{missing_item}"
                enriched.append(
                    {
                        "event_type": EVENT_TYPES[event_class],
                        "class": event_class,
                        "confidence": person["confidence"],
                        "bbox": person_box,
                        "camera_id": camera_id,
                        "zone": zone,
                        "timestamp": timestamp,
                        "person_id": index,
                    }
                )

        for item in raw_detections:
            name = item["class"].lower()
            if name == "person":
                continue
            if name in NEGATIVE_CLASSES and any(
                _contains(person["bbox"], item["bbox"]) for person in persons
            ):
                continue
            if name in GEAR_CLASSES:
                event_type = "GEAR"
            elif name in EVENT_TYPES:
                event_type = EVENT_TYPES[name]
            elif name in ("smoke", "fire"):
                event_type = name.upper()
            else:
                continue
            enriched.append(
                {
                    **item,
                    "event_type": event_type,
                    "camera_id": camera_id,
                    "zone": zone,
                    "timestamp": timestamp,
                }
            )

        return enriched, person_summaries

    def process_frame(self, frame, camera_id="camera-1", zone=None):
        self.frame_count += 1
        hazard_inference = self.frame_count == 1 or self.frame_count % self.frame_stride == 0
        started = time.perf_counter()
        camera_zones = self.config.get("camera_zones", {})
        zone = zone or camera_zones.get(camera_id, "Unassigned")

        raw = self._model_detections(self.ppe_model, frame, self.confidence, self.image_size)
        if hazard_inference:
            self.last_hazards = self.fire_smoke_detector.detect(frame)
        raw.extend(self.last_hazards)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        detections, self.last_persons = self._enrich_detections(raw, camera_id, zone, timestamp)
        elapsed = max(time.perf_counter() - started, 0.001)
        self.last_fps = 1.0 / elapsed

        annotated = frame.copy()
        counts = {"compliant": 0, "violations": 0, "smoke": 0, "fire": 0}
        for person in self.last_persons:
            counts["compliant" if person["status"] == "compliant" else "violations"] += 1

        colors = {
            "PERSON": (0, 180, 0),
            "GEAR": (0, 180, 0),
            "NO_HELMET": (0, 0, 255),
            "NO_VEST": (0, 0, 255),
            "NO_BOOTS": (0, 0, 255),
            "NO_GLOVES": (0, 0, 255),
            "SMOKE": (0, 165, 255),
            "FIRE": (0, 80, 255),
        }
        for item in detections:
            event_type = item["event_type"]
            if event_type in ("PERSON", "GEAR") and item.get("compliance") == "violation":
                color = (0, 0, 255)
            else:
                color = colors.get(event_type, (0, 180, 0))
            x1, y1, x2, y2 = item["bbox"]
            label = f"{item['class']} {item['confidence']:.2f}"
            if item.get("track_id") is not None:
                label = f"T{item['track_id']} {label}"
            if item.get("compliance"):
                label = f"{label} ({item['compliance']})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if event_type == "SMOKE":
                counts["smoke"] += 1
            elif event_type == "FIRE":
                counts["fire"] += 1

        cv2.putText(
            annotated,
            f"FPS {self.last_fps:.1f} | Compliant {counts['compliant']} | Violations {counts['violations']} | Smoke {counts['smoke']} | Fire {counts['fire']}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return annotated, detections, {
            "inference_performed": True,
            "hazard_inference_performed": hazard_inference,
            "fps": self.last_fps,
            "counts": counts,
            "persons": deepcopy(self.last_persons),
            "frame_index": self.frame_count,
        }