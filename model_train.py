import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
df = pd.read_csv('system_log.csv')

# 数据预处理
# 将时间戳转换为自第一个时间戳以来的总秒数
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

# 对 'label' 列进行编码
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 分离特征 (X) 和目标 (y)
X = df.drop('label', axis=1)
y = df['label']

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练 XGBoost 模型
# We set `use_label_encoder=False` to avoid a deprecation warning.
# `eval_metric='mlogloss'` is used for multiclass classification.
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 进行预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型的准确率为: {accuracy}")