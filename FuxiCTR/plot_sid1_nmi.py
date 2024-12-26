import torch
from matplotlib import pyplot as plt
from IPython import embed
import sys
from collections import defaultdict
import numpy as np
from matplotlib.colors import Normalize
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
plot_bar = True
plot_line = True
plot_scale_line = False
hidden = 64
width = 1

color_list = {
    f"RQ-VAE SID 1": "#52bacc",
    f"RQ-VAE Flatten": "#30688D",
    f"MOC Flatten": "#32B67B",
}
plot_data = {
    "RQ-VAE SID 1": np.array([0.0, 0.0118, 0.01251, 0.01336, 0.01388, 0.01418, 0.01458, 0.01502, 0.01536, 0.01513, 0.01567, 0.01606, 0.01643, 0.01644, 0.01661, 0.01715, 0.01727, 0.01742, 0.01759, 0.01788, 0.01776]),
    "RQ-VAE Flatten": np.array([0.0, 0.01323, 0.01322, 0.01384, 0.01291, 0.01336, 0.01432, 0.01361, 0.01456, 0.0145, 0.01473, 0.01474, 0.0151, 0.01534, 0.01442, 0.01507, 0.01444, 0.01515, 0.01567, 0.01505, 0.01572]),
    "MOC Flatten": np.array([0.0, 0.0157, 0.01575, 0.01607, 0.01598, 0.01641, 0.01704, 0.01734, 0.01741, 0.01726, 0.01702, 0.01756, 0.01852, 0.01772, 0.01815, 0.01884, 0.01887, 0.01866, 0.01875, 0.01894, 0.01897]) 
}
plt.figure(figsize=(6, 6))
for idx, k in enumerate(plot_data.keys()):
    x = [10*i for i in range(21)]
    y = np.array(plot_data[k])
    plt.plot(x, y, label=k, color=color_list[k], linewidth=2.5, markersize=5)
plt.grid()
x_ticks_list = [50*i for i in range(5)]
plt.xticks(x_ticks_list, x_ticks_list, fontsize=16)
plt.legend(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([-10,200])
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(1,2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(16)
plt.xlabel("Cluster Number", fontsize=20)
plt.ylabel("NMI", fontsize=20)
plt.savefig("./vis/sid1_nmi_line.pdf",bbox_inches='tight')
