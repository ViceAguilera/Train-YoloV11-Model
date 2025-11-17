from ultralytics import YOLO
import yaml


def main() -> None:
    """
    Entrenamiento afinado v2:
    - Modelo base: yolo11s.pt (más capacidad que yolo11n.pt).
    - Imagen: 640 px (como train12) para controlar tiempo.
    - Hiperparámetros en tuned_v2.yaml (cos_lr, augs suaves, etc.).
    """
    model_name = "yolo11s.pt"

    with open("tuned_v2.yaml", "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    model = YOLO(model_name)
    model.train(data="config.yaml", **overrides)


if __name__ == "__main__":
    main()

