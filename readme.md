#  Digit_Spirit

2025年HITSZ创芯杯项目上位机部分，使用xgboost判断用户状态，配合下位机实现更舒适的精神互动。

:warning:由于数据采集有限，因此模型聚焦于更纯净的典型场景，分别为`coding`、`video`、`gaming`、`idle`，对于更混沌的场景并未涉猎。

## 运行逻辑

<img src="readme.assets/image-20250924155426476.png" alt="image-20250924155426476" style="zoom:50%;" />

## 模型概述

​	XGBoost模型的输入为：

```python
FINAL_FEATURE_COLUMNS = [
    'cpu_percent', 'ram_percent', 'gpu_percent', 'gpu_vram_percent',
    'mouse_left_click_freq', 'mouse_right_click_freq', 'mouse_scroll_freq',
    'keyboard_counts_freq', 'mouse_distance_freq','bytes_sent_per_sec_freq', 'bytes_recv_per_sec_freq', 'packets_sent_per_sec_freq', 'packets_recv_per_sec_freq',
    'read_bytes_per_sec_freq', 'write_bytes_per_sec_freq'
]
#主要分为：CPU、GPU、RAM、VRAM、鼠标左击、鼠标右击、滚轮、鼠标移动、键盘输入频率、网络、磁盘
```

* 带freq后缀的输入为使用原始采集的数据经过10s滑窗构造而来。
* CPU、GPU、RAM、VRAM非绝对值，而是以增量的形式输入，基准为idle状态的平均值。

输出为四种状态值：`coding,video,gaming,idle`

目前认为限制模型表现主要为数据部分，因此，XGBoost的参数采用默认参数。

模型表现如下：

<img src="readme.assets/confusion_matrix.png" alt="confusion_matrix" style="zoom: 50%;" />

<img src="readme.assets/feature_importance.png" alt="feature_importance" style="zoom: 50%;" />

## 文件概述

* `model_test_ui/model_test.py`：exe的源文件
* `model_train.ipynb`：训练源文件
* `data_processs.py`：数据处理文件
* `ui_test.py`：数据采集文件