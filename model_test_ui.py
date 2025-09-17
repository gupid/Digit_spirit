import flet as ft
import threading
import time
import joblib
import pandas as pd
import asyncio
import sys
import os
from model_test import Recorder

# --- 全局配置 ---
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_path, 'xgboost_model.joblib')
ENCODER_PATH = os.path.join(base_path, 'label_encoder.joblib')
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

class StatusPredictorApp:
    def __init__(self):
        self.page = None
        self.is_running = False
        self.data_buffer = pd.DataFrame(columns=RAW_DATA_COLUMNS + ['timestamp'])

        self.idle_means = {
            'cpu_percent': 6.07, 'ram_percent': 59.58,
            'gpu_percent': 26.43, 'gpu_vram_percent': 16.29
        }

        # UI 控件
        self.status_label = ft.Text("状态: 未开始", size=14)
        self.predicted_status_label = ft.Text("--", size=32, weight=ft.FontWeight.BOLD, color="blue")
        self.info_label = ft.Text("模型和编码器待加载", size=10, color=ft.Colors.GREY)
        self.control_button = ft.ElevatedButton(text="开始监控", on_click=self.toggle_monitoring, width=150, height=50)
        self.calibrate_button = ft.ElevatedButton(text="空闲状态校准", on_click=self.start_calibration, width=150, height=50)

    async def main(self, page: ft.Page):
        self.page = page
        page.title = "用户状态实时监控"
        page.window_width = 450
        page.window_height = 350
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

        # 加载模型
        try:
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.info_label.value = "模型和编码器已加载"
        except FileNotFoundError:
            self.info_label.value = f"错误: 模型文件未找到"
            self.info_label.color = ft.colors.RED
            self.control_button.disabled = True
            self.calibrate_button.disabled = True

        self.system_monitor = Recorder()
        page.on_window_event = self.on_window_event

        # 构建页面布局
        page.add(
            ft.Column(
                [
                    self.status_label,
                    self.predicted_status_label,
                    self.info_label,
                    ft.Row(
                        [self.control_button, self.calibrate_button],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20
            )
        )
        page.update()

    async def toggle_monitoring(self, e):
        if self.is_running:
            self.is_running = False
            self.system_monitor.stop()
            self.control_button.text = "开始监控"
            self.status_label.value = "状态: 已停止"
            self.predicted_status_label.value = "--"
            self.calibrate_button.disabled = False
        else:
            self.is_running = True
            self.data_buffer = pd.DataFrame(columns=RAW_DATA_COLUMNS + ['timestamp'])
            self.system_monitor.start()
            self.control_button.text = "停止监控"
            self.status_label.value = "状态: 监控中..."
            self.calibrate_button.disabled = True
            asyncio.create_task(self.predict_loop())

        self.page.update()

    async def start_calibration(self, e):
        """启动校准过程的入口点"""
        if self.is_running:
            await self.show_dialog("提示", "请先停止监控再进行校准。")
            return
        
        # 禁用按钮防止重复点击
        self.calibrate_button.disabled = True
        self.control_button.disabled = True
        self.page.update()
        
        # 启动异步校准任务
        asyncio.create_task(self.calibrate_idle())

    async def calibrate_idle(self):
        """异步空闲状态校准"""
        print("[调试] 校准线程已启动。")
        
        # 第一步：等待5秒
        self.status_label.value = "状态: 5秒后开始校准..."
        self.page.update()
        
        for i in range(5, 0, -1):
            self.status_label.value = f"状态: {i}秒后开始校准..."
            self.page.update()
            await asyncio.sleep(1)
        
        # 第二步：开始数据采集
        self.status_label.value = "状态: 正在校准..."
        self.page.update()
        
        calib_monitor = Recorder()
        calib_monitor.start()
        
        print("[调试] 开始5秒数据采集循环...")
        collected_data = []
        for i in range(5):
            await asyncio.sleep(1)
            raw_data = calib_monitor.get_and_reset_data()
            print(f"[调试] 第 {i+1} 次采集到的原始数据: {raw_data}")
            if raw_data:
                collected_data.append(raw_data[5:])
        
        calib_monitor.stop()
        print(f"[调试] 数据采集完成。collected_data 的内容是: {collected_data}")
        
        # 第三步：处理数据并更新基准值
        if not collected_data or all(not item for item in collected_data):
            print("[调试] collected_data为空，进入校准失败逻辑。")
            self.status_label.value = "状态: 校准失败"
            await self.show_dialog("错误", "未能收集到校准数据。") # MODIFIED: Added await
        else:
            print("[调试] collected_data不为空，进入数据处理和成功逻辑。")
            df_calib = pd.DataFrame(collected_data, columns=RAW_DATA_COLUMNS[5:])
            
            print("[调试] 创建的DataFrame内容如下:")
            print(df_calib)

            self.idle_means.update({
                'cpu_percent': df_calib['cpu_percent'].mean(),
                'ram_percent': df_calib['ram_percent'].mean(),
                'gpu_percent': df_calib['gpu_percent'].mean() if 'gpu_percent' in df_calib and not df_calib['gpu_percent'].isnull().all() else -1,
                'gpu_vram_percent': df_calib['gpu_vram_percent'].mean() if 'gpu_vram_percent' in df_calib and not df_calib['gpu_vram_percent'].isnull().all() else -1
            })
            self.status_label.value = "状态: 校准完成"
            result_text = (f"校准完成！\n新基准:\n"
                           f"CPU: {self.idle_means['cpu_percent']:.2f}%\n"
                           f"RAM: {self.idle_means['ram_percent']:.2f}%\n"
                           f"GPU: {self.idle_means['gpu_percent']:.2f}%\n"
                           f"VRAM: {self.idle_means['gpu_vram_percent']:.2f}%")
            
            print("[调试] 准备显示成功对话框。")
            await self.show_dialog("成功", result_text) # MODIFIED: Added await
        
        # 重新启用按钮
        self.calibrate_button.disabled = False
        self.control_button.disabled = False
        self.page.update()

    async def predict_loop(self):
        while self.is_running:
            try:
                current_time = pd.Timestamp.now()
                raw_data = self.system_monitor.get_and_reset_data()
                new_row = pd.DataFrame([raw_data], columns=RAW_DATA_COLUMNS)
                new_row['timestamp'] = current_time

                self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
                buffer_start_time = current_time - pd.Timedelta(seconds=DATA_BUFFER_SECONDS)
                self.data_buffer = self.data_buffer[self.data_buffer['timestamp'] > buffer_start_time]

                if len(self.data_buffer) < 5:
                    self.predicted_status_label.value = f"收集中 {len(self.data_buffer)}/5"
                else:
                    window_start_time = current_time - pd.Timedelta(seconds=10)
                    recent_data = self.data_buffer[self.data_buffer['timestamp'] > window_start_time]
                    feature_vector = {
                        'mouse_distance_freq': recent_data['mouse_distance'].sum(),
                        'mouse_left_click_freq': recent_data['mouse_left_click'].sum(),
                        'mouse_right_click_freq': recent_data['mouse_right_click'].sum(),
                        'mouse_scroll_freq': recent_data['mouse_scroll'].sum(),
                        'keyboard_counts_freq': recent_data['keyboard_counts'].sum()
                    }
                    latest_resources = self.data_buffer.iloc[-1]
                    feature_vector.update({
                        'cpu_percent': latest_resources['cpu_percent'] - self.idle_means['cpu_percent'],
                        'ram_percent': latest_resources['ram_percent'] - self.idle_means['ram_percent'],
                        'gpu_percent': latest_resources['gpu_percent'] - self.idle_means['gpu_percent'],
                        'gpu_vram_percent': latest_resources['gpu_vram_percent'] - self.idle_means['gpu_vram_percent']
                    })
                    model_input = pd.DataFrame([feature_vector])[FINAL_FEATURE_COLUMNS]
                    prediction_numeric = self.model.predict(model_input)
                    prediction_label = self.label_encoder.inverse_transform(prediction_numeric)
                    self.predicted_status_label.value = f"{prediction_label[0].upper()}"
                
                self.page.update()

            except Exception as e:
                print(f"Error in predict_loop: {e}")

            await asyncio.sleep(PREDICTION_INTERVAL_MS / 1000)

    # MODIFIED: Changed back to async def
    async def show_dialog(self, title, content):
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(content),
            actions=[ft.TextButton("确定", on_click=self.close_dialog)], # MODIFIED: on_click can call async method directly
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    # MODIFIED: Changed back to async def
    async def close_dialog(self, e):
        # The dialog is automatically passed to the event handler as e.control
        dialog = e.control.page.dialog
        dialog.open = False
        self.page.update()

    async def on_window_event(self, e):
        if e.data == "close":
            if self.is_running:
                self.is_running = False
                self.system_monitor.stop()
            self.page.window_destroy()


if __name__ == "__main__":
    app = StatusPredictorApp()
    ft.app(target=app.main)