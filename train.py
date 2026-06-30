from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/Users/mehmetdegirmenci/Desktop/Microsoft proje/yolo_dataset_small/data.yaml",
    epochs=20,
    imgsz=320,
    batch=16,
    device="mps",
    project="/Users/mehmetdegirmenci/Desktop/Microsoft proje/runs",
    name="uav_detector",
    patience=10,
    save_period=5,
    val=True,  # val aslında validation dosyasını kullanacak mı kullanmayacak mı 
)
