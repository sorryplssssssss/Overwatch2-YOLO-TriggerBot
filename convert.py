import asyncio
import os
from ultralytics.data.converter import convert_ndjson_to_yolo

# Используем ТОЧНОЕ имя файла, которое показал Python
ndjson_file = 'overwatch.yolov11.ndjson'
output_dir = 'datasets'

async def main():
    if not os.path.exists(ndjson_file):
        print(f"❌ Ошибка: файл {ndjson_file} всё ещё не найден!")
        return

    print(f"✅ Файл {ndjson_file} найден! Начинаю конвертацию...")
    try:
        # Запускаем асинхронную конвертацию
        yaml_path = await convert_ndjson_to_yolo(ndjson_file, output_path=output_dir)
        print(f"\n✅ Конвертация прошла успешно! Файл конфигурации: {yaml_path}")
    except Exception as e:
        print(f"\n❌ Ошибка внутри Ultralytics: {e}")

if __name__ == "__main__":
    asyncio.run(main())