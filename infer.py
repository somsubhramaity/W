import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPE YOLO inference on an image or video source.")
    parser.add_argument(
        "--source",
        default="datasets/construction-ppe/images/test",
        help="Image, video, directory, webcam index, or RTSP URL",
    )
    parser.add_argument("--model", default="models/ppe_yolov8.pt", help="PPE YOLO weights")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output-dir", default="runs/detect")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"PPE model not found: {model_path}")

    source = args.source.strip()
    if source.isdecimal():
        source = int(source)
    elif not source.lower().startswith(("rtsp://", "http://", "https://")):
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path if (Path.cwd() / source_path).exists() else ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Inference source not found: {source_path}")
        source = str(source_path)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.predict(
        source=source,
        conf=args.confidence,
        imgsz=min(args.image_size, 640),
        save=True,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
    )
    print(f"Inference done. Results saved to {output_dir / args.name}")


if __name__ == "__main__":
    main()
