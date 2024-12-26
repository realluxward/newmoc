import torch
from matplotlib import pyplot as plt
from IPython import embed
import sys
from collections import defaultdict
import numpy as np
from matplotlib.colors import Normalize
data_list = {
    "MOC": np.array([0.01329, 0.01339, 0.01269, 0.01357, 0.01644, 0.01596, 0.01702]),
    "RQ-VAE": np.array([0.01329, 0.01393, 0.01437, 0.01344, 0.01394, 0.01395, 0.01473]),
    "ME": np.array([0.01329, 0.01427, 0.01227, 0.01377, 0.01145, 0.01211, 0.01356])
}
# colors = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#1B9E8A', '#32B67B', '#6CCC5F', '#B4DD3D', '#FDE73A']
color_list = {
    f"ME": "#482976",
    f"RQ-VAE": "#30688D",
    f"MOC": "#32B67B",
}
plt.figure(figsize=(6, 6))
for idx, k in enumerate(data_list.keys()):
    x = np.arange(1,8)
    y = np.array(data_list[k])
    plt.plot(x, y, label=k, color=color_list[k], linewidth=2, markersize=8)
plt.grid()
plt.xticks(x, [f'{i}x' for i in range(1,8)], fontsize=16)
plt.legend(fontsize=16)
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(1,2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(16)
plt.yticks(fontsize=16)
# plt.ylim([0.005, 0.075])

plt.ylabel("NMI", fontsize=16)
plt.xlabel("Scaling Factor", fontsize=16)
plt.savefig(f"./vis/scale_line.pdf",bbox_inches='tight')
