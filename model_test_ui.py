import flet as ft
import asyncio
import sys
import os
import joblib
import pandas as pd
from model_test import Recorder
import win32gui

# --- 全局配置 ---
# 兼容打包后的路径
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_path, 'xgboost_model.joblib')
ENCODER_PATH = os.path.join(base_path, 'label_encoder.joblib')
CSV_LABEL_PATH = os.path.join(base_path, 'windows_label.csv')
PREDICTION_INTERVAL_MS = 1000
DATA_BUFFER_SECONDS = 30

# 特征列，确保与模型训练时一致
FINAL_FEATURE_COLUMNS = [
    'cpu_percent', 'ram_percent', 'gpu_percent', 'gpu_vram_percent',
    'mouse_left_click_freq', 'mouse_right_click_freq', 'mouse_scroll_freq',
    'keyboard_counts_freq', 'mouse_distance_freq'
]

# 原始数据列
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

        # 默认的空闲状态基准值
        self.idle_means = {
            'cpu_percent': 6.0, 'ram_percent': 60.0,
            'gpu_percent': 25.0, 'gpu_vram_percent': 15.0
        }

        # --- UI 控件定义 ---
        self.status_label = ft.Text("状态: 未开始", size=14)
        self.predicted_status_label = ft.Text("--", size=32, weight=ft.FontWeight.BOLD, color="blue")
        self.current_window_label = ft.Text(
            "当前窗口: --", size=12, color=ft.colors.GREY_500,
            italic=True, no_wrap=True, tooltip="当前检测到的前景窗口标题"
        )
        self.info_label = ft.Text("模型和编码器待加载", size=10, color=ft.colors.GREY)
        self.control_button = ft.ElevatedButton(text="开始监控", on_click=self.toggle_monitoring, width=150, height=50)
        self.calibrate_button = ft.ElevatedButton(text="空闲状态校准", on_click=self.start_calibration, width=150, height=50)
        
        # --- 字典管理UI控件 ---
        self.dict_view = ft.ListView(expand=1, spacing=5, auto_scroll=True)
        self.dict_key_input = ft.TextField(label="窗口标题关键词", width=220)
        self.dict_value_input = ft.TextField(label="对应标签", width=120)
        self.add_button = ft.ElevatedButton("添加/更新", icon=ft.icons.ADD, on_click=self.add_or_update_entry)
        
        try:
            self.windows_dictionary = pd.read_csv(CSV_LABEL_PATH).set_index('title')['label'].to_dict()
        except FileNotFoundError:
            self.windows_dictionary = {}
            self.info_label.value = f"警告: '{os.path.basename(CSV_LABEL_PATH)}' 未找到"
            self.info_label.color = ft.colors.ORANGE

    async def main(self, page: ft.Page):
        self.page = page
        page.title = "用户状态实时监控"
        page.window_width = 550
        page.window_height = 550
        page.theme = ft.Theme(font_family="Microsoft YaHei")
        page.dark_theme = ft.Theme(font_family="Microsoft YaHei")
        page.on_window_event = self.on_window_event
        
        # 加载模型
        try:
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.info_label.value = "模型和编码器已加载"
        except FileNotFoundError:
            self.info_label.value = "错误: 模型或编码器文件未找到"
            self.info_label.color = ft.colors.RED
            self.control_button.disabled = True
            self.calibrate_button.disabled = True

        self.system_monitor = Recorder()
        
        # 创建UI布局
        self._build_ui()
        
        # 首次加载时填充字典视图
        await self.update_dict_view()

    def _build_ui(self):
        """构建UI界面"""
        tab_monitor = ft.Tab(
            text="实时监控", icon=ft.icons.VIDEOCAM,
            content=ft.Container(
                content=ft.Column(
                    [
                        self.status_label,
                        self.predicted_status_label,
                        self.current_window_label,
                        self.info_label,
                        ft.Row(
                            [self.control_button, self.calibrate_button],
                            alignment=ft.MainAxisAlignment.CENTER, spacing=20
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20
                ),
                alignment=ft.alignment.center, padding=20
            )
        )

        tab_dictionary = ft.Tab(
            text="字典管理", icon=ft.icons.BOOK,
            content=ft.Column(
                [
                    ft.Text("添加或修改字典条目:", weight=ft.FontWeight.BOLD),
                    ft.Row([self.dict_key_input, self.dict_value_input], alignment=ft.MainAxisAlignment.CENTER),
                    self.add_button,
                    ft.Divider(),
                    ft.Text("当前字典内容:", weight=ft.FontWeight.BOLD),
                    self.dict_view,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10,
                scroll=ft.ScrollMode.ADAPTIVE
            )
        )

        self.page.add(
            ft.Tabs(selected_index=0, animation_duration=300, tabs=[tab_monitor, tab_dictionary], expand=1)
        )

    async def toggle_monitoring(self, e):
        if self.is_running:
            self.is_running = False
            self.system_monitor.stop()
            self.control_button.text = "开始监控"
            self.status_label.value = "状态: 已停止"
            self.predicted_status_label.value = "--"
            self.current_window_label.value = "当前窗口: --"
            self.calibrate_button.disabled = False
        else:
            self.is_running = True
            self.data_buffer = pd.DataFrame(columns=RAW_DATA_COLUMNS + ['timestamp'])
            self.system_monitor.start()
            self.control_button.text = "停止监控"
            self.status_label.value = "状态: 监控中..."
            self.calibrate_button.disabled = True
            asyncio.create_task(self.predict_loop())
        await self.page.update_async()

    async def start_calibration(self, e):
        if self.is_running:
            await self.show_dialog("提示", "请先停止监控再进行校准。")
            return
        
        self.calibrate_button.disabled = True
        self.control_button.disabled = True
        await self.page.update_async()
        
        asyncio.create_task(self.calibrate_idle())

    async def calibrate_idle(self):
        for i in range(5, 0, -1):
            self.status_label.value = f"状态: {i}秒后开始校准..."
            await self.page.update_async()
            await asyncio.sleep(1)
        
        self.status_label.value = "状态: 正在校准...请保持空闲"
        await self.page.update_async()
        
        calib_monitor = Recorder()
        calib_monitor.start()
        
        collected_data = []
        for _ in range(5):
            await asyncio.sleep(1)
            raw_data = calib_monitor.get_and_reset_data()
            if raw_data:
                collected_data.append(raw_data[5:]) # 只取资源使用率部分
        
        calib_monitor.stop()
        
        if not collected_data or all(not item for item in collected_data):
            self.status_label.value = "状态: 校准失败"
            await self.show_dialog("错误", "未能收集到校准数据。")
        else:
            df_calib = pd.DataFrame(collected_data, columns=RAW_DATA_COLUMNS[5:])
            self.idle_means.update({
                'cpu_percent': df_calib['cpu_percent'].mean(),
                'ram_percent': df_calib['ram_percent'].mean(),
                'gpu_percent': df_calib['gpu_percent'].mean() if 'gpu_percent' in df_calib and not df_calib['gpu_percent'].isnull().all() else -1,
                'gpu_vram_percent': df_calib['gpu_vram_percent'].mean() if 'gpu_vram_percent' in df_calib and not df_calib['gpu_vram_percent'].isnull().all() else -1
            })
            self.status_label.value = "状态: 校准完成"
            result_text = (f"校准完成！\n新基准:\n"
                           f"CPU: {self.idle_means['cpu_percent']:.2f}%, "
                           f"RAM: {self.idle_means['ram_percent']:.2f}%\n"
                           f"GPU: {self.idle_means['gpu_percent']:.2f}%, "
                           f"VRAM: {self.idle_means['gpu_vram_percent']:.2f}%")
            await self.show_dialog("成功", result_text)
        
        self.calibrate_button.disabled = False
        self.control_button.disabled = False
        await self.page.update_async()

    def _get_window_title_info(self):
        """获取前景窗口标题并更新UI"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                window_title = win32gui.GetWindowText(hwnd)
                display_title = (window_title[:45] + '...') if len(window_title) > 45 else window_title
                self.current_window_label.value = f"当前窗口: {display_title}"
                self.current_window_label.tooltip = f"完整标题: {window_title}"
                return window_title
            else:
                self.current_window_label.value = "当前窗口: 无"
                return ""
        except Exception:
             self.current_window_label.value = "当前窗口: 获取失败"
             return ""


    async def predict_loop(self):
        """主预测循环，采用“空闲优先 -> 字典规则 -> 模型兜底”逻辑"""
        while self.is_running:
            try:
                # 步骤 0: 更新当前窗口标题信息
                window_title = self._get_window_title_info()

                # 步骤 1: 数据采集与缓冲
                raw_data = self.system_monitor.get_and_reset_data()
                
                if raw_data is None:
                    await asyncio.sleep(PREDICTION_INTERVAL_MS / 1000)
                    continue

                current_time = pd.Timestamp.now()
                new_row = pd.DataFrame([raw_data], columns=RAW_DATA_COLUMNS)
                new_row['timestamp'] = current_time
                self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
                self.data_buffer = self.data_buffer[self.data_buffer['timestamp'] > (current_time - pd.Timedelta(seconds=DATA_BUFFER_SECONDS))]

                if len(self.data_buffer) < 5:
                    self.predicted_status_label.value = f"收集中 {len(self.data_buffer)}/5"
                    await self.page.update_async()
                    await asyncio.sleep(PREDICTION_INTERVAL_MS / 1000)
                    continue

                # 步骤 2: 计算特征并进行模型预测
                recent_data = self.data_buffer[self.data_buffer['timestamp'] > (current_time - pd.Timedelta(seconds=10))]
                latest_resources = self.data_buffer.iloc[-1]
                
                feature_vector = {
                    'mouse_distance_freq': recent_data['mouse_distance'].sum(),
                    'mouse_left_click_freq': recent_data['mouse_left_click'].sum(),
                    'mouse_right_click_freq': recent_data['mouse_right_click'].sum(),
                    'mouse_scroll_freq': recent_data['mouse_scroll'].sum(),
                    'keyboard_counts_freq': recent_data['keyboard_counts'].sum()
                }
                
                # [逻辑修复] 处理校准值为-1的情况
                feature_vector.update({
                    'cpu_percent': latest_resources['cpu_percent'] - self.idle_means['cpu_percent'],
                    'ram_percent': latest_resources['ram_percent'] - self.idle_means['ram_percent'],
                    'gpu_percent': latest_resources['gpu_percent'] - self.idle_means['gpu_percent'] if self.idle_means['gpu_percent'] != -1 else latest_resources['gpu_percent'],
                    'gpu_vram_percent': latest_resources['gpu_vram_percent'] - self.idle_means['gpu_vram_percent'] if self.idle_means['gpu_vram_percent'] != -1 else latest_resources['gpu_vram_percent']
                })

                model_input = pd.DataFrame([feature_vector])[FINAL_FEATURE_COLUMNS]
                prediction_numeric = self.model.predict(model_input)
                model_prediction = self.label_encoder.inverse_transform(prediction_numeric)[0]

                final_prediction = ""
                # 步骤 3: 决策 - 空闲状态优先
                if model_prediction == 'idle':
                    final_prediction = "IDLE"
                else:
                    # 步骤 4: 决策 - 字典规则
                    dict_label = None
                    for key_title, label in self.windows_dictionary.items():
                        if key_title.lower() in window_title.lower():
                            dict_label = label
                            break
                    
                    if dict_label:
                        final_prediction = dict_label
                    else:
                        # 步骤 5: 决策 - 模型兜底
                        final_prediction = model_prediction
                
                # 更新UI
                self.predicted_status_label.value = final_prediction.upper()
                await self.page.update_async()

            except Exception as e:
                print(f"Error in predict_loop: {e}")
                self.predicted_status_label.value = "错误"
                await self.page.update_async()

            await asyncio.sleep(PREDICTION_INTERVAL_MS / 1000)

    async def show_dialog(self, title, content):
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text(title), content=ft.Text(content),
            actions=[ft.TextButton("确定", on_click=self.close_dialog)],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog = dialog
        dialog.open = True
        await self.page.update_async()

    async def close_dialog(self, e):
        self.page.dialog.open = False
        await self.page.update_async()

    async def on_window_event(self, e):
        if e.data == "close":
            if self.is_running:
                self.is_running = False
                self.system_monitor.stop()
            self.page.window_destroy()

    async def update_dict_view(self):
        self.dict_view.controls.clear()
        for key, label in sorted(self.windows_dictionary.items()):
            self.dict_view.controls.append(
                ft.Row([
                    ft.IconButton(icon=ft.icons.DELETE_FOREVER, icon_color="red400",
                                  tooltip="删除此条目", data=key, on_click=self.delete_entry),
                    ft.Text(f"标题含: '{key}'", weight=ft.FontWeight.BOLD),
                    ft.Text(f" -> 标签: {label}"),
                ], alignment=ft.MainAxisAlignment.START)
            )
        await self.page.update_async()

    async def save_dict_to_csv(self):
        df_to_save = pd.DataFrame(list(self.windows_dictionary.items()), columns=['title', 'label'])
        try:
            df_to_save.to_csv(CSV_LABEL_PATH, index=False)
        except Exception as e:
            print(f"保存字典失败: {e}")
            await self.show_dialog("错误", f"无法保存字典文件:\n{e}")

    async def add_or_update_entry(self, e):
        key = self.dict_key_input.value.strip()
        value = self.dict_value_input.value.strip()

        if not key or not value:
            self.dict_key_input.error_text = "关键词不能为空" if not key else None
            self.dict_value_input.error_text = "标签不能为空" if not value else None
            await self.page.update_async()
            return

        self.dict_key_input.error_text, self.dict_value_input.error_text = None, None
        self.windows_dictionary[key] = value
        self.dict_key_input.value, self.dict_value_input.value = "", ""
        
        await self.save_dict_to_csv()
        await self.update_dict_view()

    async def delete_entry(self, e):
        key_to_delete = e.control.data
        if key_to_delete in self.windows_dictionary:
            del self.windows_dictionary[key_to_delete]
            await self.save_dict_to_csv()
            await self.update_dict_view()

if __name__ == "__main__":
    app = StatusPredictorApp()
    ft.app(target=app.main)