import mss
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import pyautogui
import pydirectinput

pydirectinput.PAUSE = 0.01 
pydirectinput.FAILSAFE = False

print("Загрузка TensorRT...", flush=True)
model = YOLO("best.engine")
print("Модель загружена! Ожидание цели...", flush=True)

w, h = pyautogui.size()
desiredHeight = 600
desiredWidth  = 400
top = (h // 2) - (desiredHeight // 2)
left = (w // 2) - (desiredWidth // 2)

center_x = desiredWidth // 2
center_y = desiredHeight // 2

last_click_time = 0
CLICK_COOLDOWN = 0.15  # Задержка между выстрелами

with mss.mss() as sct:
    while True:
        img = sct.grab((left, top, left + desiredWidth, top + desiredHeight))
        img_np = np.array(img)
        img_rgb = cv.cvtColor(img_np, cv.COLOR_RGBA2RGB)

        results = model(img_rgb, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        # Создаем картинку для отладки
        debug_img = img_rgb.copy()
        
        # Рисуем красный квадрат в центре (это наша зона поражения)
        cv.rectangle(debug_img, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), (0, 0, 255), 2)

        if len(boxes) > 0:
            for box in boxes:
                b = box.xyxy[0]
                x_min, y_min, x_max, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                
                # Рисуем зеленую рамку врага
                cv.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

                # УВЕЛИЧИЛИ ОТСТУП ДО 40 ПИКСЕЛЕЙ (чтобы прощало промахи из-за задержки)
                margin = 40
                
                # Если красный квадрат пересекается с зеленой рамкой (даже с отступом)
                if (x_min - margin) < center_x < (x_max + margin) and (y_min - margin) < center_y < (y_max + margin):
                    current_time = time.time()
                    
                    if current_time - last_click_time >= CLICK_COOLDOWN:
                        print(f"🎯 ВЫСТРЕЛ!", flush=True)
                        cv.rectangle(debug_img, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), (0, 255, 255), 2) # Квадрат желтеет при выстреле
                        
                        pydirectinput.mouseDown(button='left')
                        time.sleep(0.01)
                        pydirectinput.mouseUp(button='left')
                            
                        last_click_time = current_time

        # Показываем окно отладки
        cv.imshow("Bot Vision", debug_img)
        
        # Нажмите 'q' в окне бота, чтобы закрыть его
        if cv.waitKey(1) & 0xFF == ord('q'):
            break