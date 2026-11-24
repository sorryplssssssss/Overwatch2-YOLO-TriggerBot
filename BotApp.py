import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2 as cv
import numpy as np
import mss
from ultralytics import YOLO
import pyautogui
import pydirectinput
import math

pydirectinput.PAUSE = 0.01
pydirectinput.FAILSAFE = False

class TriggerBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Overwatch TriggerBot (Fast MSS)")
        self.geometry("350x450+20+20")
        self.configure(bg="#1e1e1e")

        self.bot_running = False
        self.bot_thread = None
        self.stop_event = threading.Event()
        self.model = None

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

        ttk.Label(self, text="Overwatch TriggerBot", font=("Segoe UI", 14, "bold")).pack(pady=10)

        self.status_var = tk.StringVar(value="Статус: Остановлен")
        self.status_label = ttk.Label(self, textvariable=self.status_var, foreground="#ff5555")
        self.status_label.pack(pady=2)

        self.detect_var = tk.StringVar(value="Целей в кадре: 0")
        ttk.Label(self, textvariable=self.detect_var, foreground="#55ffff").pack(pady=2)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        ttk.Label(self, textvariable=self.fps_var).pack(pady=4)

        ttk.Label(self, text="Порог уверенности:").pack(pady=(10, 0))
        self.conf_var = tk.DoubleVar(value=0.15)
        ttk.Scale(self, from_=0.05, to=0.9, variable=self.conf_var, orient="horizontal", length=200).pack()

        ttk.Label(self, text="Радиус зоны поражения (px):").pack(pady=(10, 0))
        self.radius_var = tk.IntVar(value=80)
        ttk.Scale(self, from_=20, to=150, variable=self.radius_var, orient="horizontal", length=200).pack()

        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Окно Bot Vision", variable=self.debug_var).pack(pady=10)

        self.btn_start = ttk.Button(self, text="▶ ЗАПУСТИТЬ БОТА", command=self.toggle_bot)
        self.btn_start.pack(pady=10, fill="x", padx=40)

    def toggle_bot(self):
        if not self.bot_running:
            self.start_bot()
        else:
            self.stop_bot()

    def start_bot(self):
        self.bot_running = True
        self.stop_event.clear()
        self.status_var.set("Загрузка модели...")
        self.status_label.configure(foreground="#ffaa00")
        self.btn_start.configure(text="⏹ ОСТАНОВИТЬ")

        self.bot_thread = threading.Thread(target=self.bot_loop, daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        self.bot_running = False
        self.stop_event.set()
        self.status_var.set("Статус: Остановлен")
        self.status_label.configure(foreground="#ff5555")
        self.btn_start.configure(text="▶ ЗАПУСТИТЬ БОТА")
        cv.destroyAllWindows()

    def bot_loop(self):
        try:
            if self.model is None:
                # Загружаем быструю TensorRT модель
                self.model = YOLO("v2.engine")

            self.status_var.set("Работает (Fast MSS)")
            self.status_label.configure(foreground="#55ff55")

            w, h = pyautogui.size()
            desiredHeight, desiredWidth = 600, 400
            top = (h // 2) - (desiredHeight // 2)
            left = (w // 2) - (desiredWidth // 2)

            center_x, center_y = desiredWidth // 2, desiredHeight // 2
            last_click_time = 0
            
            last_target_time = 0
            last_target_cx = 0
            last_target_cy = 0
            MEMORY_TIME = 0.2

            frame_count = 0
            start_time = time.time()

            cv.namedWindow("Bot Vision", cv.WINDOW_NORMAL)
            cv.setWindowProperty("Bot Vision", cv.WND_PROP_TOPMOST, 1)

            # ИСПОЛЬЗУЕМ НАДЕЖНЫЙ MSS
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    img = sct.grab((left, top, left + desiredWidth, top + desiredHeight))
                    img_np = np.array(img)
                    img_bgr = np.ascontiguousarray(img_np[:, :, :3])

                    # ИНФЕРЕНС: imgsz=416 ускоряет обработку
                    results = self.model(img_bgr, conf=self.conf_var.get(), imgsz=416, verbose=False)
                    boxes = results[0].boxes
                    
                    debug_img = img_bgr.copy() if self.debug_var.get() else None
                    is_shooting = False
                    current_time = time.time()

                    if len(boxes) > 0:
                        self.detect_var.set(f"Целей в кадре: {len(boxes)}")
                        closest_dist = 9999
                        best_box = None

                        for box in boxes:
                            b = box.xyxy[0]
                            x_min, y_min, x_max, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                            enemy_cx = (x_min + x_max) / 2
                            enemy_cy = (y_min + y_max) / 2
                            distance = math.hypot(enemy_cx - center_x, enemy_cy - center_y)

                            if debug_img is not None:
                                cv.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                            if distance < closest_dist:
                                closest_dist = distance
                                best_box = (enemy_cx, enemy_cy, distance)

                        if best_box is not None:
                            last_target_cx, last_target_cy, dist = best_box
                            last_target_time = current_time
                            
                            if dist < self.radius_var.get():
                                is_shooting = True

                    else:
                        self.detect_var.set("Целей в кадре: 0")
                        if current_time - last_target_time < MEMORY_TIME:
                            dist = math.hypot(last_target_cx - center_x, last_target_cy - center_y)
                            if dist < self.radius_var.get():
                                is_shooting = True
                                if debug_img is not None:
                                    cv.circle(debug_img, (int(last_target_cx), int(last_target_cy)), 10, (0, 255, 0), 2)

                    if is_shooting:
                        if current_time - last_click_time >= 0.12:
                            pydirectinput.mouseDown(button='left')
                            time.sleep(0.03)
                            pydirectinput.mouseUp(button='left')
                            last_click_time = current_time

                    frame_count += 1
                    if frame_count >= 20:
                        fps = frame_count / (time.time() - start_time)
                        self.fps_var.set(f"FPS: {fps:.1f}")
                        frame_count = 0
                        start_time = time.time()

                    if self.debug_var.get():
                        circle_color = (0, 255, 255) if is_shooting else (0, 0, 255)
                        cv.circle(debug_img, (center_x, center_y), self.radius_var.get(), circle_color, 2)
                        cv.imshow("Bot Vision", debug_img)
                        
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            self.stop_bot()
                            break

        except Exception as e:
            # Если произойдет ошибка, она покажется прямо в окне программы
            self.status_var.set(f"Ошибка: {str(e)[:30]}...")
            self.status_label.configure(foreground="#ff0000")
            self.stop_bot()

if __name__ == "__main__":
    app = TriggerBotApp()
    app.mainloop()