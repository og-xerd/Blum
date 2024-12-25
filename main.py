import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import pygetwindow
import time
import os
import mouse
import keyboard
import random

dev = False

def load_model() -> tuple[YOLO, list]:
    os.system("cls")
    print("Model loading...")
    model = YOLO("best.pt")
    os.system("cls")
    print("Model loaded!")

    names = model.names
    model.to("cpu")
    return model, names

model, names = load_model()

def get_window():
    try:
        windows = pygetwindow.getWindowsWithTitle("TelegramDesktop")

        window = windows[0]
        if not window.isActive:
            window.minimize()
            window.restore()
        return {
            "height": window.height,
            "left": window.left,
            "top": window.top, 
            "width": window.width,
        }
    except:
        print("Error: TelegramDesktop window not found.")
        os._exit(0)

window = get_window()
def grab_screenshot(window):
    with mss() as sct:
        img = sct.grab(window)
        screenshot = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot_rgb


os.system("cls")
print("""
______________ _____  _______  ___
___  __ )__  / __  / / /__   |/  /
__  __  |_  /  _  / / /__  /|_/ / 
_  /_/ /_  /___/ /_/ / _  /  / /  
/_____/ /_____/\____/  /_/  /_/   
                                  
""")
print("Press 'enter' to start or 's' to stop!")
active = False
while True:
    if active == True:
        try:
            screenshot = grab_screenshot(window)

            results = model.predict(screenshot, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf = round(box.conf.item(), 2)
                    name = names[int(box.cls)]

                    if name == "bomb":
                        continue

                    if name == "snowman" or conf > 0.8:
                        if name == "next_button":
                            random_time = random.uniform(1.0, 5.0)
                            print("Next button detected, waiting for", random_time, "seconds")
                            time.sleep(random_time)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = window["left"] + (x1 + x2) // 2, window["top"] + (y1 + y2) // 2
                        mouse.move(cx, cy, absolute=True)
                        mouse.click()

            for result in results:
                annotated_frame = result.plot()

            if dev:
                cv2.imshow("YOLO Screenshot", annotated_frame)

            cv2.waitKey(33)

            if keyboard.is_pressed('s'):
                print("Stopping...")
                active = False

        except Exception as e:
            print(f"Błąd podczas predykcji: {e}")
            break
    else:
        if keyboard.is_pressed('enter'):
            window = get_window()
            print("Starting...")
            active = True

        time.sleep(0.01)
