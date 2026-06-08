from pipeline.detect import run_detection


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path> [model_path]")
        raise SystemExit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"
    run_detection(sys.argv[1], model_path)
