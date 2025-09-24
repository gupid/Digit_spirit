import pandas as pd

def process_dataframe(df):
    columns_to_process = ['cpu_percent', 'ram_percent', 'gpu_percent', 'gpu_vram_percent']
    for col in columns_to_process:
        # 检查列是否存在于DataFrame中，避免出错
        if col in df.columns:
            df[col] -= df.loc[df['label']=='idle',col].mean()
    return df

# 1. 将所有需要处理的文件路径放入一个列表
file_paths = [
    #'train_data/system_log_9_5.csv',
    #'train_data/system_log_9_8.csv',
    #'train_data/system_log_9_9.csv',
    #'train_data/system_log_9_13.csv',
    'train_data/system_log_9_22.csv',
    #'train_data/system_log_9_23.csv',
    'train_data/system_log_9_24.csv',
]

file_paths_test = [
    'test_data/system_log_9_23.csv',
]
# 2. 使用列表推导式和循环来读取和处理所有文件
processed_dfs = [process_dataframe(pd.read_csv(file)) for file in file_paths]

processed_df_test = [process_dataframe(pd.read_csv(file) )for file in file_paths_test]

# 3. 合并所有处理好的DataFrame
combined_df = pd.concat(processed_dfs, ignore_index=True)
combined_df_test = pd.concat(processed_df_test, ignore_index=True)
# 4. 保存结果
combined_df.to_csv('system_log.csv', index=False)
combined_df_test.to_csv('system_log_test.csv', index=False)
print("数据处理完成，并已保存到 system_log.csv/system_log_test.csv")