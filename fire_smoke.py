"""Fire and smoke detection using the community YOLOv8 checkpoint."""

from pathlib import Path


REPO_ID = "rabahdev/fire-smoke-yolov8n"
MODEL_FILENAME = "best.pt"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "fire_smoke"
CLASS_NAMES = {0: "smoke", 1: "fire"}


def download_weights():
    """Download once into the repository so subsequent runs can work offline."""
    cached_path = MODEL_DIR / MODEL_FILENAME
    if cached_path.is_file():
        return cached_path

    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(MODEL_DIR),
        )
    )


class FireSmokeDetector:
    def __init__(self, model_path=None, confidence=0.25, image_size=640):
        from ultralytics import YOLO

        candidate = Path(model_path) if model_path else None
        weights = candidate if candidate and candidate.is_file() else download_weights()
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        detections = []
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES.get(class_id, str(class_id))
                detections.append(
                    {
                        "class": class_name,
                        "event_type": class_name.upper(),
                        "confidence": float(box.conf[0]),
                        "bbox": tuple(int(value) for value in box.xyxy[0]),
                    }
                )
        return detections