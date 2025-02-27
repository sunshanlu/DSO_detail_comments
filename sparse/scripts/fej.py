import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 定义x和y的范围及分辨率
x = np.linspace(0, 2, 500)
y = np.linspace(0, 2, 500)
X, Y = np.meshgrid(x, y)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 定义x和y的范围及分辨率
x = np.linspace(0, 2, 500)
y = np.linspace(0, 2, 500)
X, Y = np.meshgrid(x, y)

# 定义颜色列表：从深蓝到淡蓝再到浅绿、翠绿、黄色、艳红色到深红色
# colors = ["#00028c", "#000df6", "#89ff74", "#ff211e", "#850008"]
colors = ["#00028c", "#89ff74", "#ff211e", "#850008"]

# 创建自定义颜色映射
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 计算函数值（注意表达式简化后的形式）
Z = (X * Y - 1) ** 2

# 将Z值大于2的部分设为2
Z[Z > 1.5] = 1.5

# 绘制热力图
plt.figure(figsize=(10, 8))
heatmap = plt.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
plt.grid(True, alpha=0.3)

# 禁用x轴和y轴的刻度缩放
plt.xticks(np.arange(0, 2.5, 0.5))
plt.yticks(np.arange(0, 2.5, 0.5))

# 显示图形
plt.show()
