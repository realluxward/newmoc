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
    f"w/o MOC Fusion": "#30688D",
    f"w/ MOC Fusion": "#32B67B",
}
plot_data = {
    "w/o MOC Fusion": np.array([0.0, 0.00828, 0.00878, 0.0091, 0.01066, 0.00989, 0.0113, 0.01258, 0.01179, 0.01268, 0.01356, 0.01374, 0.01421, 0.01451, 0.0151, 0.01535, 0.0158, 0.01559, 0.01652, 0.01666, 0.01651]),
    "w/ MOC Fusion": np.array([0.0, 0.0157, 0.01575, 0.01607, 0.01598, 0.01641, 0.01704, 0.01734, 0.01741, 0.01726, 0.01702, 0.01756, 0.01852, 0.01772, 0.01815, 0.01884, 0.01887, 0.01866, 0.01875, 0.01894, 0.01897]) 
}
plt.figure(figsize=(8, 6))
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
plt.savefig("./vis/mix_cluster.pdf",bbox_inches='tight')
