import tkinter as tk
from tkinter import messagebox
import threading
import time
import joblib
import pandas as pd
from model_test import Recorder

# --- 全局配置 ---
MODEL_PATH = 'xgboost_model.joblib'
ENCODER_PATH = 'label_encoder.joblib'
PREDICTION_INTERVAL_MS = 1000
DATA_BUFFER_SECONDS = 30

FINAL_FEATURE_COLUMNS = [
    'cpu_percent', 'ram_percent', 'gpu_percent', 'gpu_vram_percent', 
    'mouse_left_click_freq', 'mouse_right_click_freq', 'mouse_scroll_freq', 
    'keyboard_counts_freq', 'mouse_distance_freq'
]

RAW_DATA_COLUMNS = [
    'mouse_distance', 'mouse_left_click', 'mouse_right_click',
    'mouse_scroll', 'keyboard_counts', 'cpu_percent',
    'ram_percent', 'gpu_percent', 'gpu_vram_percent'
]


class StatusPredictorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("用户状态实时监控")
        self.geometry("450x300")

        try:
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
        except FileNotFoundError:
            messagebox.showerror("错误", f"模型文件 ({MODEL_PATH} 或 {ENCODER_PATH}) 未找到。")
            self.destroy()
            return

        self.system_monitor = Recorder()
        self.is_running = False
        self.data_buffer = pd.DataFrame(columns=RAW_DATA_COLUMNS + ['timestamp'])
        
        self.idle_means = {
            'cpu_percent': 6.07,
            'ram_percent': 59.58,
            'gpu_percent': 26.43,
            'gpu_vram_percent': 16.29
        }

        self.status_label = tk.Label(self, text="状态: 未开始", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        self.predicted_status_label = tk.Label(self, text="--", font=("Helvetica", 28, "bold"), fg="blue")
        self.predicted_status_label.pack(pady=10)
        self.info_label = tk.Label(self, text="模型和编码器已加载", font=("Helvetica", 10), fg="gray")
        self.info_label.pack(pady=5)
        
        button_frame = tk.Frame(self)
        button_frame.pack(pady=20)
        self.control_button = tk.Button(button_frame, text="开始监控", command=self.toggle_monitoring, width=15, height=2)
        self.control_button.pack(side=tk.LEFT, padx=10)
        self.calibrate_button = tk.Button(button_frame, text="空闲状态校准", command=self.start_calibration_thread, width=15, height=2)
        self.calibrate_button.pack(side=tk.LEFT, padx=10)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_monitoring(self):
        if self.is_running:
            self.is_running = False
            self.system_monitor.stop()
            self.control_button.config(text="开始监控")
            self.status_label.config(text="状态: 已停止")
            self.predicted_status_label.config(text="--")
            self.calibrate_button.config(state=tk.NORMAL)
        else:
            self.is_running = True
            self.data_buffer = pd.DataFrame(columns=RAW_DATA_COLUMNS + ['timestamp'])
            self.system_monitor.start()
            self.control_button.config(text="停止监控")
            self.status_label.config(text="状态: 监控中...")
            self.calibrate_button.config(state=tk.DISABLED)
            self.predict_loop()

    def start_calibration_thread(self):
        calibration_thread = threading.Thread(target=self.calibrate_idle, daemon=True)
        calibration_thread.start()

    def calibrate_idle(self):
        if self.is_running:
            messagebox.showwarning("提示", "请先停止监控再进行校准。")
            return

        self.calibrate_button.config(state=tk.DISABLED)
        self.control_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 5秒后开始校准...")
        messagebox.showinfo("校准开始", "将在5秒后开始收集空闲数据，请保持电脑完全静止5秒钟。")
        time.sleep(5) 
        self.status_label.config(text="状态: 正在校准...")
        
        calib_monitor = Recorder()
        calib_monitor.start()
        
        collected_data = []
        for _ in range(5):
            time.sleep(1)
            raw_data = calib_monitor.get_and_reset_data()
            collected_data.append(raw_data[5:])
        
        calib_monitor.stop()
        
        if not collected_data:
            messagebox.showerror("错误", "未能收集到校准数据。")
            self.status_label.config(text="状态: 校准失败")
            return
            
        df_calib = pd.DataFrame(collected_data, columns=RAW_DATA_COLUMNS[5:])
        self.idle_means['cpu_percent'] = df_calib['cpu_percent'].mean()
        self.idle_means['ram_percent'] = df_calib['ram_percent'].mean()
        self.idle_means['gpu_percent'] = df_calib['gpu_percent'].mean() if 'gpu_percent' in df_calib else -1
        self.idle_means['gpu_vram_percent'] = df_calib['gpu_vram_percent'].mean() if 'gpu_vram_percent' in df_calib else -1
        
        self.status_label.config(text="状态: 校准完成")
        messagebox.showinfo("成功", f"校准完成！\n新基准:\n"
                                  f"CPU: {self.idle_means['cpu_percent']:.2f}%\n"
                                  f"RAM: {self.idle_means['ram_percent']:.2f}%\n"
                                  f"GPU: {self.idle_means['gpu_percent']:.2f}%\n"
                                  f"VRAM: {self.idle_means['gpu_vram_percent']:.2f}%")
        
        self.calibrate_button.config(state=tk.NORMAL)
        self.control_button.config(state=tk.NORMAL)

    def predict_loop(self):
        if not self.is_running:
            return

        try:
            current_time = pd.Timestamp.now()
            raw_data = self.system_monitor.get_and_reset_data()
            
            new_row = pd.DataFrame([raw_data], columns=RAW_DATA_COLUMNS)
            new_row['timestamp'] = current_time
            
            # 【代码优化】修复FutureWarning，让代码更稳健
            if self.data_buffer.empty:
                self.data_buffer = new_row
            else:
                self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
            
            buffer_start_time = current_time - pd.Timedelta(seconds=DATA_BUFFER_SECONDS)
            self.data_buffer = self.data_buffer[self.data_buffer['timestamp'] > buffer_start_time]

            if len(self.data_buffer) < 5: 
                self.predicted_status_label.config(text=f"收集中 {len(self.data_buffer)}/5")
                # 这里不再需要 return，因为 finally 会调度下一次执行
            else:
                window_start_time = current_time - pd.Timedelta(seconds=10)
                recent_data = self.data_buffer[self.data_buffer['timestamp'] > window_start_time]
                
                feature_vector = {}
                feature_vector['mouse_distance_freq'] = recent_data['mouse_distance'].sum()
                feature_vector['mouse_left_click_freq'] = recent_data['mouse_left_click'].sum()
                feature_vector['mouse_right_click_freq'] = recent_data['mouse_right_click'].sum()
                feature_vector['mouse_scroll_freq'] = recent_data['mouse_scroll'].sum()
                feature_vector['keyboard_counts_freq'] = recent_data['keyboard_counts'].sum()
                
                latest_resources = self.data_buffer.iloc[-1]
                feature_vector['cpu_percent'] = latest_resources['cpu_percent'] - self.idle_means['cpu_percent']
                feature_vector['ram_percent'] = latest_resources['ram_percent'] - self.idle_means['ram_percent']
                feature_vector['gpu_percent'] = latest_resources['gpu_percent'] - self.idle_means['gpu_percent']
                feature_vector['gpu_vram_percent'] = latest_resources['gpu_vram_percent'] - self.idle_means['gpu_vram_percent']
                
                model_input = pd.DataFrame([feature_vector])
                model_input = model_input[FINAL_FEATURE_COLUMNS]
                
                prediction_numeric = self.model.predict(model_input)
                prediction_label = self.label_encoder.inverse_transform(prediction_numeric)
                
                self.predicted_status_label.config(text=f"{prediction_label[0].upper()}")

        except Exception as e:
            print(f"Error in predict_loop: {e}")
        finally:
            self.after(PREDICTION_INTERVAL_MS, self.predict_loop)
        
    def on_closing(self):
        if self.is_running:
            self.is_running = False
            self.system_monitor.stop()
        self.destroy()

if __name__ == "__main__":
    app = StatusPredictorUI()
    app.mainloop()