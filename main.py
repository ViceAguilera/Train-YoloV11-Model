from ultralytics import YOLO

# if __name__ == '__main__':
#    model = YOLO("yolo11n.pt")
#    model.train(
#        data="config.yaml", 
#        epochs=100, 
#        imgsz=640, 
#        batch=16, 
#        project="runs",
#        augment=True,
#        mosaic=1.0,
#        mixup=0.2,
#        copy_paste=0.2,
#        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#        erasing=0.4,
#        auto_augment="simple",
#        optimizer="SGD",
#        lr0=0.01,
#        lrf=0.001,
#        cos_lr=True,
#        warmup_epochs=5.0,
#        weight_decay=5e-4,
#        rect=True,
#        multi_scale=True,
#        patience=50,
#        save_period=10
#        )
    
    
if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.train(data="config.yaml", epochs=150, imgsz=640, batch=16, project="runs")