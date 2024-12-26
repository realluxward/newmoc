import matplotlib.pyplot as plt
import numpy as np

# 数据点
x = np.arange(7)  # 单位：M
rq_data = np.array([0.7447,	0.7446,	0.7443,	0.7444,	0.744,	0.7441,	0.7441,])  # 单位：未知
moc_data = np.array([0.7449,	0.7446,	0.7445,	0.7447,	0.745,	0.7443,	0.7446,])

cmap = plt.get_cmap('viridis').colors

# 创建图表
cmap = plt.get_cmap('viridis').colors

# 创建图表
plt.figure(figsize=(8, 6))

color_list = {
    'rq':cmap[10],
    'moc':cmap[150],    
}

# 绘制散点图，使用颜色映射
plt.plot(x, rq_data, c=color_list['rq'], marker='o', linestyle='--', markersize=10, linewidth=2.5, label='RQ-VAE')
plt.plot(x, moc_data, c=color_list['moc'], marker='o', markersize=10, linewidth=2.5, linestyle='--', label='MOC')

# 添加标题和标签
plt.xticks(x,[f"{i}" for i in range(1,8)], fontsize=24)
plt.xlabel('Semantic ID index', fontsize=24)
plt.ylabel('Test AUC', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim([0.743, 0.748])

# 显示网格
plt.grid(True)

plt.legend(fontsize=24)

# 显示图表
plt.savefig("./vis/uni_line.pdf",  bbox_inches='tight')