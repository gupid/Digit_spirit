import threading
import time
from pynput import keyboard , mouse
import psutil

def thread_on_click_cb(x,y,button,pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with {button}")
    else:
        print(f"Mouse released at ({x}, {y}) with {button}")

def thread_on_press_cb(key):
    try:
        print(f"Key {key.char} pressed")
    except AttributeError:
        print(f"Special key {key} pressed")
def thread_cpu_ram_cb():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")

if __name__ == "__main__":
    mouse_listener = mouse.Listener(on_click=thread_on_click_cb)
    keyboard_listener = keyboard.Listener(on_press=thread_on_press_cb)
    mouse_listener.start()
    keyboard_listener.start()

    t = threading.Thread(target=thread_cpu_ram_cb)
    t.start()
    keyboard_listener.join()
    mouse_listener.join()
    t.join()
# ...existing code...