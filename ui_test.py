import tkinter as tk
import tkinter.font as tkfont
import threading
import time
import psutil
from pynput import keyboard, mouse
import os
import csv
import datetime
import pynvml
import collections # 导入collections模块以使用deque

class Recorder:
    """
    一个用于记录系统状态和用户输入的类。
    它在后台线程中运行，以避免阻塞GUI。
    """
    def __init__(self, label_var, status_var):
        self.label_var = label_var
        self.status_var = status_var
        
        # 线程和监听器
        self.mouse_listener = None
        self.keyboard_listener = None
        self.stats_thread = None 
        self.stop_event = threading.Event()
        
        # 状态与配置
        self.running = False
        self.output_filename = "train_data/system_log_9_24.csv"
        
        # 数据采集变量
        self.mouse_locations = []
        self.mouse_left_clicks = 0
        self.mouse_right_clicks = 0
        self.mouse_scroll_amount = 0
        self.keyboard_counts = 0
        
        self.data_buffer = collections.deque(maxlen=5)
        
        # 性能与同步
        self.data_lock = threading.Lock()
        self.last_move_time = 0
        self.throttle_time = 0.1

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

    def system_stats_worker(self):
        """后台线程，每秒采集一次数据并写入文件。"""
        # 使用'a'模式打开文件，确保在程序多次启动和停止时不会覆盖旧数据
        try:
            file = open(self.output_filename, 'a', newline='', encoding='utf-8')
            writer = csv.writer(file)
        except IOError as e:
            error_msg = f"错误: 无法打开文件 {self.output_filename}"
            self.status_var.set(error_msg)
            print(f"{error_msg}. Reason: {e}")
            return # 如果文件无法打开，则终止工作线程

        while not self.stop_event.is_set():
            # 1. 采集系统数据 (为了避免等待1秒，我们将interval设为None，然后在循环末尾sleep)
            cpu_usage = psutil.cpu_percent(interval=None) 
            ram_usage = psutil.virtual_memory().percent
            gpu_usage, gpu_vram_usage = -1, -1
            if self.gpu_handle:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_usage = gpu_util.gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_vram_usage = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
                except pynvml.NVMLError as e:
                    print(f"Could not get GPU info: {e}")

            # 2. 采集用户输入数据
            with self.data_lock:
                mouse_distance_sum = sum(((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5 for p1, p2 in zip(self.mouse_locations, self.mouse_locations[1:]))
                left_clicks = self.mouse_left_clicks
                right_clicks = self.mouse_right_clicks
                scroll_amount = self.mouse_scroll_amount
                keyboard_hits = self.keyboard_counts
                
                # 重置数据
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

            # 3. 准备数据行
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_label = self.label_var.get()
            data_row = [
                timestamp, mouse_distance_sum, left_clicks, right_clicks, scroll_amount, 
                keyboard_hits, cpu_usage, ram_usage, gpu_usage, gpu_vram_usage,bytes_sent_per_sec, bytes_recv_per_sec, packets_sent_per_sec, packets_recv_per_sec,
                read_bytes_per_sec, write_bytes_per_sec,
                current_label
            ]
            
            # 如果缓冲区已满 (5条记录)，则将最旧的记录写入文件
            if len(self.data_buffer) == self.data_buffer.maxlen:
                oldest_data = self.data_buffer[0] # 获取最旧的数据
                writer.writerow(oldest_data)
                self.status_var.set(f"数据已记录于 {oldest_data[0]}")
                print(f"Data recorded: {oldest_data}")

            # 将新数据添加到缓冲区的右侧
            self.data_buffer.append(data_row)
            
            # 等待1秒钟，完成每秒采集一次的循环
            time.sleep(1)

        print("Flushing remaining data from buffer...")
        for row in self.data_buffer:
            writer.writerow(row)
            print(f"Flushed data: {row}")
        self.data_buffer.clear()
        file.close() # 关闭文件

    def _delayed_start_worker(self):
        """在后台等待5秒，然后启动监听器和数据采集线程。"""
        # 倒计时
        for i in range(5, 0, -1):
            # 检查在倒计时期间用户是否取消了操作
            if not self.running:
                print("Recording cancelled during countdown.")
                self.status_var.set("记录已取消")
                return
            self.status_var.set(f"{i}秒后开始记录...")
            time.sleep(1)
        
        # 再次检查，以防在最后1秒内取消
        if not self.running:
            print("Recording cancelled just before start.")
            self.status_var.set("记录已取消")
            return

        self.status_var.set(f"开始记录 '{self.label_var.get()}'...")
        print(f"Recorder started with label '{self.label_var.get()}'")

        # 启动监听器
        self.mouse_listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move, on_scroll=self.on_scroll)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()

        #网络、磁盘
        self.net_io = psutil.net_io_counters()
        self.disk_io = psutil.disk_io_counters()
        
        # 启动数据采集线程
        self.stats_thread = threading.Thread(target=self.system_stats_worker, daemon=True)
        self.stats_thread.start()

    def start(self):
        if self.running: return
        self.running = True
        
        # 确保CSV文件头存在
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'mouse_distance', 'mouse_left_click', 'mouse_right_click', 'mouse_scroll',
                    'keyboard_counts','cpu_percent', 'ram_percent', 'gpu_percent','gpu_vram_percent','bytes_sent_per_sec', 'bytes_recv_per_sec', 'packets_sent_per_sec', 'packets_recv_per_sec',
                    'read_bytes_per_sec', 'write_bytes_per_sec',
                    'label'
                ])
        
        self.stop_event.clear()
        self.data_buffer.clear() # 每次开始前清空缓冲区
        
        # 启动一个新线程来处理5秒的延迟，以避免阻塞GUI
        delay_thread = threading.Thread(target=self._delayed_start_worker, daemon=True)
        delay_thread.start()

    def stop(self):
        if not self.running: return
        self.running = False # 立即设置状态，这可以让延迟启动的倒计时中断
        self.status_var.set("正在停止记录...")

        self.stop_event.set()
        if self.mouse_listener: self.mouse_listener.stop()
        if self.keyboard_listener: self.keyboard_listener.stop()
        if self.stats_thread: self.stats_thread.join(timeout=2) # 等待线程将缓冲区数据写完
        
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
                print("NVML shut down.")
            except pynvml.NVMLError as e:
                print(f"Error shutting down NVML: {e}")
        
        self.status_var.set("记录已停止")
        print("Recorder stopped")

    def toggle_recording(self):
        if self.running:
            self.stop()
        else:
            self.start()
        return self.running

def on_resize(event):
    base = min(event.width, event.height)
    new_font_size = max(8, int(base / 20))
    btn_font.configure(size=new_font_size)
    on_off_button.config(padx=max(5, int(event.width * 0.05)), pady=max(5, int(event.height * 0.05)))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("数据采集程序")
    root.geometry("500x350")

    btn_font = tkfont.Font(family="Arial", size=16, weight="bold")
    
    selected_label = tk.StringVar(value="coding")
    label_frame = tk.Frame(root)
    label_frame.pack(pady=20)
    tk.Label(label_frame, text="选择标签:").pack(side="left", padx=(0, 10))
    tk.Radiobutton(label_frame, text="编程", variable=selected_label, value="coding").pack(side="left")
    tk.Radiobutton(label_frame, text="游戏", variable=selected_label, value="gaming").pack(side="left")
    tk.Radiobutton(label_frame, text="浏览", variable=selected_label, value="browsing").pack(side="left")
    tk.Radiobutton(label_frame, text="视频", variable=selected_label, value="video").pack(side="left")
    tk.Radiobutton(label_frame, text="闲置", variable=selected_label, value="idle").pack(side="left")

    status_var = tk.StringVar(value="准备就绪")
    status_bar = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor="w", bd=1, padx=5)
    status_bar.pack(side="bottom", fill="x")

    recorder = Recorder(selected_label, status_var)

    def switch_record_cb():
        on_off_button.config(state="disabled")
        root.update_idletasks()
        is_running = recorder.toggle_recording()
        on_off_button.config(text="结束记录" if is_running else "开始记录")
        on_off_button.config(state="normal")

    on_off_button = tk.Button(root, text="开始记录", command=switch_record_cb, font=btn_font, bg="#4CAF50", fg="white")
    on_off_button.pack(fill="both", expand=True, padx=20, pady=10)
    
    root.bind("<Configure>", on_resize)

    def on_close():
        recorder.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    print("窗口已关闭")