import tkinter as tk
import tkinter.font as tkfont
import threading
import time
import psutil
from pynput import keyboard, mouse
import os
import csv
import datetime
# 新增：导入 pynvml 模块
import pynvml

record_on_flag = False

def switch_record_cb():
    global record_on_flag
    record_on_flag = not record_on_flag
    
    if record_on_flag:
        on_off_button.config(text="结束")
    else:
        on_off_button.config(text="开始")
        
    print("record_on_flag =", record_on_flag)


class Recorder:
    def __init__(self, label_var):
        self.label_var = label_var
        self.mouse_listener = None
        self.keyboard_listener = None
        self.stats_thread = None # 修改：线程变量重命名
        self.stop_event = threading.Event()
        self.running = False
        self.output_filename = "system_log.csv"
        self.gpu_handle = None
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("NVIDIA GPU found and initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize NVIDIA GPU monitoring. Error: {e}")

    def on_click(self, x, y, button, pressed):
        current_label = self.label_var.get()
        if pressed:
            print(f"[{current_label}] Mouse clicked at ({x}, {y}) with {button}")
        else:
            print(f"[{current_label}] Mouse released at ({x}, {y}) with {button}")

    def on_press(self, key):
        current_label = self.label_var.get()
        try:
            print(f"[{current_label}] Key {key.char} pressed")
        except AttributeError:
            print(f"[{current_label}] Special key {key} pressed")

    def system_stats_worker(self):
        while not self.stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            
            gpu_usage = -1 # 默认值-1，代表未获取到
            if self.gpu_handle:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_usage = gpu_util.gpu
                except Exception as e:
                    print(f"Could not get GPU usage: {e}")

            # 准备要写入文件的数据
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_label = self.label_var.get()
            data_row = [timestamp, cpu_usage, ram_usage, gpu_usage, current_label]
            
            # 使用'a'模式（append）来不覆盖地追加写入
            with open(self.output_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
            
            print(f"Data recorded: {data_row}")

    def start(self):
        if self.running:
            return
        
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'cpu_percent', 'ram_percent', 'gpu_percent','label'])

        self.stop_event.clear()
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.stats_thread = threading.Thread(target=self.system_stats_worker, daemon=True)
        self.stats_thread.start()
        self.running = True
        print(f"Recorder started with label '{self.label_var.get()}'")

    def stop(self):
        if not self.running:
            return
        self.stop_event.set()
        
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
                print("NVML shut down.")
            except:
                pass

        if self.mouse_listener is not None:
            try:
                self.mouse_listener.stop()
            except Exception:
                pass
            self.mouse_listener = None
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
            self.keyboard_listener = None
        self.running = False
        print("Recorder stopped")


def on_resize(event):
    base = min(event.width, event.height)
    new_font_size = max(8, int(base / 15))
    btn_font.configure(size=new_font_size)
    on_off_button.config(padx=max(4, int(event.width * 0.03)),
                        pady=max(2, int(event.height * 0.02)))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("数据采集程序")
    root.geometry("400x300")

    btn_font = tkfont.Font(family="Segoe UI", size=12, weight="normal")
    
    selected_label = tk.StringVar(value="coding")

    label_frame = tk.Frame(root)
    label_frame.pack(pady=10)

    tk.Label(label_frame, text="选择标签:").pack(side="left", padx=(0, 10))

    coding_radio = tk.Radiobutton(label_frame, text="Coding", variable=selected_label, value="coding")
    gaming_radio = tk.Radiobutton(label_frame, text="Gaming", variable=selected_label, value="gaming")
    coding_radio.pack(side="left")
    gaming_radio.pack(side="left")

    on_off_button = tk.Button(root, text="开始", command=switch_record_cb, font=btn_font)
    on_off_button.pack(fill="both", expand=True, padx=20, pady=(0, 20))
   
    root.bind("<Configure>", on_resize)

    recorder = Recorder(selected_label)

    def monitor_record():
        if record_on_flag:
            recorder.start()
        else:
            recorder.stop()
        root.after(200, monitor_record)

    def on_close():
        recorder.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(200, monitor_record)
    root.mainloop()

    print("窗口已关闭")