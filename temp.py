import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 从文件中读取CSV数据
file_path = (
    "/mnt/g/suanleme/vsmaf/workernodes.csv"  # 将'your_file.csv'替换为你的CSV文件名
)
df = pd.read_csv(file_path)

# 计算每个GPU型号的数量
gpu_counts = df["gpu_name"].value_counts()

# 只显示前n个频率最高的GPU型号，其他的归为“其他”
top_n = 19
top_gpus = gpu_counts.head(top_n)
other_gpus = gpu_counts[top_n:].sum()
gpu_counts_top_n = top_gpus._append(pd.Series(other_gpus, index=["others"]))

# 生成随机颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(gpu_counts_top_n)))

# 定义标签，显示显卡型号和数量
labels = [
    f"{name} ({count})" for name, count in zip(gpu_counts_top_n.index, gpu_counts_top_n)
]

# 绘制饼图
plt.figure(figsize=(9, 9))
plt.pie(
    gpu_counts_top_n,
    labels=labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
)
plt.title(
    f"GPU Distribution on Suanleme ({datetime.now().strftime('%Y-%m-%d')})", pad=40
)  # 增加pad参数
plt.axis("equal")
plt.show()
