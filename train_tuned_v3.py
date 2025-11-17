from ultralytics import YOLO
import yaml


def main() -> None:
    """
    Entrenamiento afinado v3:
    - Modelo base: yolo11s.pt (más capacidad que yolo11n.pt).
    - Imagen: 640 px (como train12) para controlar tiempo.
    - Hiperparámetros en tuned_v3.yaml (SGD, cos_lr, augs suaves).
    """
    model_name = "yolo11s.pt"

    with open("tuned_v3.yaml", "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    model = YOLO(model_name)
    model.train(data="config.yaml", **overrides)


if __name__ == "__main__":
    main()

