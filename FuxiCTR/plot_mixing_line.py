import matplotlib.pyplot as plt
import numpy as np

# 数据点
x = np.arange(7)  # 单位：M
nomix_data = np.array([0.7449,	0.7449,	0.7453,	0.7464,	0.7468,	0.7467,	0.7468,])  # 单位：未知
moc_data = np.array([0.7449,	0.7462,	0.7458,	0.7467,	0.7472,	0.7472,	0.7474,])

cmap = plt.get_cmap('viridis').colors

# 创建图表
plt.figure(figsize=(8, 6))

color_list = {
    'rq':cmap[10],
    'moc':cmap[150],    
}

# 绘制散点图，使用颜色映射
plt.plot(x, nomix_data, c=color_list['rq'], marker='o', linestyle='--', markersize=10, linewidth=2.5, label='w/o MOC Mixing')
plt.plot(x, moc_data, c=color_list['moc'], marker='o', markersize=10, linewidth=2.5, linestyle='--', label='w/ MOC Mixing')

# 添加标题和标签
plt.xticks(x,[f"{i}x" for i in range(1,8)], fontsize=24)
plt.xlabel('Scaling Factor', fontsize=24)
plt.ylabel('Test AUC', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim([0.7430, 0.748])

# 显示网格
plt.grid(True)

plt.legend(fontsize=24)

# 显示图表
plt.savefig("./vis/mixing_line.pdf",  bbox_inches='tight')