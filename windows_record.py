import time
import win32gui
import win32process
import psutil # 需要额外安装 pip install psutil
import pandas as pd

def get_active_window_info():
    """获取最上层活动窗口的信息"""
    try:
        # 获取前景窗口的句柄 (handle)
        hwnd = win32gui.GetForegroundWindow()

        # 获取窗口标题
        window_title = win32gui.GetWindowText(hwnd)

        # 获取窗口所属进程ID
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        # 根据进程ID获取进程名称
        process_name = psutil.Process(pid).name()

        return {
            "handle": hwnd,
            "title": window_title,
            "pid": pid,
            "process_name": process_name
        }
    except Exception as e:
        # 如果没有前景窗口 (例如桌面)
        return None
    
name_of_windows = pd.DataFrame(columns=["handle", "title", "pid", "process_name"])

# 调用函数并打印信息
for i in range(100):
    active_window = get_active_window_info()
    if active_window:
        print("最上层的窗口信息:")
        print(f"  标题: {active_window['title']}")
        print(f"  进程名: {active_window['process_name']}")
        print(f"  进程ID: {active_window['pid']}")
        print(f"  窗口句柄: {active_window['handle']}")
        name_of_windows = pd.concat([name_of_windows, pd.DataFrame([active_window])], ignore_index=True)
    else:
        print("无法获取活动窗口信息。")
    time.sleep(1)  # 每2秒获取一次

name_of_windows.to_csv("name_of_windows.csv", index=False)
print("窗口信息已保存到 name_of_windows.csv")