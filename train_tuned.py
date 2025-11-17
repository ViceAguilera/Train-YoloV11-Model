from ultralytics import YOLO
import yaml


def main():
    # Use a slightly larger model for better accuracy; adjust if GPU memory is tight.
    model_name = "yolo11s.pt"  # previous runs used yolo11n.pt

    with open("tuned.yaml", "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    model = YOLO(model_name)
    # Train with tuned overrides; dataset is defined in config.yaml
    model.train(data="config.yaml", **overrides)


if __name__ == "__main__":
    main()

