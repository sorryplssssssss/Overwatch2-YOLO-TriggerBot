from ultralytics import YOLO

if __name__ == '__main__':
    # Загружаем базовую модель
    model = YOLO('yolo11n.pt')

    # Запускаем обучение
    results = model.train(
        data=r'datasets\overwatch.yolov11-646ac3f8\data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0
    )