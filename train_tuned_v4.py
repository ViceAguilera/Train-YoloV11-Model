from ultralytics import YOLO
import yaml


def main() -> None:
    """
    Entrenamiento afinado v4 con dataset Construction-PPE:
    - Modelo base: yolo11s.pt (equilibrio capacidad/velocidad).
    - Imagen: 640 px (estable para 8 GB).
    - Dataset: construction-ppe/data.yaml.
    - Hiperparámetros: derivados de tuned_v3 (SGD + cos_lr + augs suaves).
    """
    model_name = "yolo11s.pt"

    with open("tuned_v4.yaml", "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    model = YOLO(model_name)
    # Usamos el data.yaml del dataset Construction-PPE
    model.train(data="construction-ppe/data.yaml", **overrides)


if __name__ == "__main__":
    main()

