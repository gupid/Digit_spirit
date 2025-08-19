import tkinter as tk
import tkinter.font as tkfont
import threading
import time
import psutil
from pynput import keyboard, mouse

record_on_flag = False

# 定义点击按钮时要执行的函数
def switch_record_cb():
    global record_on_flag
    record_on_flag = not record_on_flag
    print("record_on_flag =", record_on_flag)

# 新增：记录器管理类，负责启动/停止监听器与后台线程
class Recorder:
    def __init__(self):
        self.mouse_listener = None
        self.keyboard_listener = None
        self.cpu_thread = None
        self.stop_event = threading.Event()
        self.running = False

    # 鼠标回调（在独立线程中由 pynput 调用）
    def on_click(self, x, y, button, pressed):
        if pressed:
            print(f"Mouse clicked at ({x}, {y}) with {button}")
        else:
            print(f"Mouse released at ({x}, {y}) with {button}")

    # 键盘回调
    def on_press(self, key):
        try:
            print(f"Key {key.char} pressed")
        except AttributeError:
            print(f"Special key {key} pressed")

    # CPU/RAM 监控线程
    def cpu_worker(self):
        # 使用阻塞式 psutil.cpu_percent(interval=1) 可以每秒采样一次
        while not self.stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            print(f"CPU Usage: {cpu_usage}%, RAM Usage: {ram_usage}%")
            # 如果需要更频繁/更精细控制，可以用短间隔 + 时间累积

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        # 创建并启动 pynput 监听器（它们内部自己运行线程）
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        # 启动后台 CPU/RAM 线程（设置为 daemon 使程序退出时自动结束）
        self.cpu_thread = threading.Thread(target=self.cpu_worker, daemon=True)
        self.cpu_thread.start()
        self.running = True
        print("Recorder started")

    def stop(self):
        if not self.running:
            return
        # 请求线程退出
        self.stop_event.set()
        # 停止 pynput 监听器
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

# 4. 绑定窗口大小变化事件，根据窗口尺寸调整字体大小和内边距
def on_resize(event):
    # 使用窗口较小边长来控制字体，防止字体过大
    base = min(event.width, event.height)
    new_font_size = max(8, int(base / 15))  # 15 是经验值，可调整
    btn_font.configure(size=new_font_size)

    # 可选：同时按比例调整按钮内部填充，使视觉更协调
    hello_button.config(padx=max(4, int(event.width * 0.03)),
                        pady=max(2, int(event.height * 0.02)))

if __name__ == "__main__":
    # 1. 创建主窗口
    root = tk.Tk()
    root.title("数据采集程序")
    root.geometry("400x300")  # 设置初始窗口大小

    # 创建可调节的字体（初始大小可设置为 12）
    btn_font = tkfont.Font(family="Segoe UI", size=12, weight="normal")

    # 2. 创建一个按钮控件，使用可调整字体
    hello_button = tk.Button(root, text="开始/结束", command=switch_record_cb, font=btn_font)

    # 3. 使用 pack() 将按钮放入窗口，并使其随窗口放大/缩小
    hello_button.pack(fill="both", expand=True, padx=20, pady=20)
   
    root.bind("<Configure>", on_resize)

    # --- 新增：使用 root.after 定期检查标志并执行动作 ---
    recorder = Recorder()

    def monitor_record():
        # 根据标志启动或停止 recorder；非阻塞，通过 after 调度
        if record_on_flag:
            recorder.start()
        else:
            recorder.stop()
        # 每 200 毫秒检查一次；按需调整间隔
        root.after(200, monitor_record)

    # 在关闭窗口时确保清理资源
    def on_close():
        recorder.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # 启动轮询（在 mainloop 运行前安排第一次调用）
    root.after(200, monitor_record)
    # --- 新增结束 ---

    # 5. 启动事件循环
    root.mainloop()

    # 这行代码在窗口关闭前不会被执行
    print("窗口已关闭")
