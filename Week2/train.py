from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=25, batch=16, imgsz=608)
