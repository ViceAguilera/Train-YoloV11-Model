from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.train(data="config.yaml", epochs=150, imgsz=640, batch=16, project="runs")