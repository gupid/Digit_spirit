import joblib
import pandas as pd
import psutil
import time
import threading
from pynput import mouse, keyboard
import pynvml

class Recorder:
    def __init__(self):
        # 监听器
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # 状态
        self.running = False
        
        # 数据采集变量
        self.mouse_locations = []
        self.mouse_left_clicks = 0
        self.mouse_right_clicks = 0
        self.mouse_scroll_amount = 0
        self.keyboard_counts = 0
        
        # 性能与同步
        self.data_lock = threading.Lock()
        self.last_move_time = 0
        self.throttle_time = 0.1 # 鼠标移动事件节流

        # 网络与磁盘
        self.bytes_sent_prev = 0
        self.bytes_recv_prev = 0
        self.packets_sent_prev = 0
        self.packets_recv_prev = 0
        self.read_bytes_prev = 0
        self.write_bytes_prev = 0

        # GPU 初始化
        self.gpu_handle = None
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("NVIDIA GPU found and initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize NVIDIA GPU monitoring. Error: {e}")

    def on_click(self, x, y, button, pressed):
        if pressed:
            with self.data_lock:
                if button == mouse.Button.left:
                    self.mouse_left_clicks += 1
                elif button == mouse.Button.right:
                    self.mouse_right_clicks += 1

    def on_move(self, x, y):
        current_time = time.time()
        if current_time - self.last_move_time < self.throttle_time:
            return
        self.last_move_time = current_time
        with self.data_lock:
            self.mouse_locations.append((x, y))

    def on_press(self, key):
        with self.data_lock:
            self.keyboard_counts += 1

    def on_scroll(self, x, y, dx, dy):
        with self.data_lock:
            self.mouse_scroll_amount += abs(dy)
    
    def get_and_reset_data(self):
        # 1. 采集系统性能数据
        cpu_usage = psutil.cpu_percent(interval=None) 
        ram_usage = psutil.virtual_memory().percent
        gpu_usage, gpu_vram_usage = -1, -1 # 默认为-1，表示不可用
        if self.gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_usage = gpu_util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_vram_usage = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            except pynvml.NVMLError as e:
                print(f"Could not get GPU info: {e}")

        # 2. 采集并重置用户输入数据
        with self.data_lock:
            # 计算鼠标移动总距离
            mouse_distance_sum = sum(((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5 for p1, p2 in zip(self.mouse_locations, self.mouse_locations[1:]))
            
            # 复制当前数据
            left_clicks = self.mouse_left_clicks
            right_clicks = self.mouse_right_clicks
            scroll_amount = self.mouse_scroll_amount
            keyboard_hits = self.keyboard_counts
            
            # 重置数据，为下一个时间窗口做准备
            self.mouse_locations.clear()
            self.mouse_left_clicks = 0
            self.mouse_right_clicks = 0
            self.mouse_scroll_amount = 0
            self.keyboard_counts = 0

        net_io_now = psutil.net_io_counters()
        disk_io_now = psutil.disk_io_counters()
        
        if self.bytes_recv_prev == 0:
            bytes_sent_per_sec = 0
            bytes_recv_per_sec = 0
            packets_sent_per_sec = 0
            packets_recv_per_sec = 0
            read_bytes_per_sec = 0
            write_bytes_per_sec = 0
        else:
            bytes_sent_per_sec = net_io_now.bytes_sent - self.bytes_sent_prev
            bytes_recv_per_sec = net_io_now.bytes_recv - self.bytes_recv_prev
            packets_sent_per_sec = net_io_now.packets_sent - self.packets_sent_prev
            packets_recv_per_sec = net_io_now.packets_recv - self.packets_recv_prev
            read_bytes_per_sec = disk_io_now.read_bytes - self.read_bytes_prev
            write_bytes_per_sec = disk_io_now.write_bytes - self.write_bytes_prev        
        self.bytes_sent_prev = net_io_now.bytes_sent
        self.bytes_recv_prev = net_io_now.bytes_recv
        self.packets_sent_prev = net_io_now.packets_sent
        self.packets_recv_prev = net_io_now.packets_recv
        self.read_bytes_prev = disk_io_now.read_bytes
        self.write_bytes_prev = disk_io_now.write_bytes

        # 3. 组合成数据行并返回
        return [
            mouse_distance_sum, left_clicks, right_clicks, scroll_amount, 
            keyboard_hits, cpu_usage, ram_usage, gpu_usage, gpu_vram_usage,bytes_sent_per_sec, bytes_recv_per_sec, packets_sent_per_sec, packets_recv_per_sec,
            read_bytes_per_sec, write_bytes_per_sec,
        ]

    def start(self):
        if self.running: return
        self.running = True

        self.get_and_reset_data()

        self.mouse_listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move, on_scroll=self.on_scroll)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        print("User input listeners started.")

    def stop(self):
        if not self.running: return
        self.running = False
        
        if self.mouse_listener: self.mouse_listener.stop()
        if self.keyboard_listener: self.keyboard_listener.stop()
        
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
                print("NVML shut down.")
            except pynvml.NVMLError as e:
                print(f"Error shutting down NVML: {e}")
        
        print("Recorder stopped.")
