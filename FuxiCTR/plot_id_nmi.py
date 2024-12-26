import torch
from matplotlib import pyplot as plt
from IPython import embed
from collections import defaultdict
import sys
import numpy as np
from matplotlib.colors import Normalize
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
plot_bar = True
plot_line = False
hidden = 64
width = 1
cmap = plt.get_cmap('tab20').colors
div=0.07015


color_list = {
    "RQ-VAE": "#35c40e",
    "ME": "#00cbbf",
    "MOC": "#ff704c",
}



plot_data = {
    "ME": np.array([0.0045911, 0.0045911 ,0.0045911 ,0.0045911 ,0.0045911 ,0.0045911 ,0.0045911]),
    "RQ-VAE": np.array([0.00641622, 0.00172412 ,0.00104498 ,0.00079535 ,0.00087605 ,0.0007879
 ,0.00079703]),
    "MOC": np.array([0.0045911,  0.00543277 ,0.00517547 ,0.00483134 ,0.0038996  ,0.00488394
 ,0.00402055]),
}

total_data = np.stack([plot_data["ME"], plot_data["RQ-VAE"], plot_data["MOC"]], axis=0)
plot_keys = [f"SID {i}" for i in range(1,8)]

plt.figure(figsize=(14, 8))
cmap = plt.get_cmap('Blues')
scale = 0.008
norm = Normalize(vmin=total_data.min()-scale, vmax=total_data.max()+scale-0.002)


for i in range(len(plot_keys)):
    x = np.arange(i, i+27, 9)
    color = norm(total_data[:, i])
    plt.bar(x, total_data[:, i], width=width, label=plot_keys[i], color=cmap(color))

plt.text(3, -0.001, f"ME", ha='center', va='bottom', fontsize=20, color='black')

plt.text(11, -0.001, f"RQ-VAE", ha='center', va='bottom', fontsize=20, color='black')

plt.text(21, -0.001, f"MOC", ha='center', va='bottom', fontsize=20, color='black')


# for cur_x, cur_y in zip(x, base_data):
#     plt.text(cur_x, cur_y+0.0003, f"{cur_y:.4f}", ha='center', va='bottom', fontsize=20, color='black')

# for cur_x, cur_y in zip(x, nomix_data):
#     plt.text(cur_x, cur_y-0.0003, f"{cur_y:.4f}", ha='center', va='top', fontsize=20, color='white')

# plt.xticks(x,['item feat', 'flatten', 'final'], fontsize=20)
# plt.xticks([])
# plt.xlabel("Index", fontsize=20)
plt.ylabel("NMI", fontsize=20)
plt.yticks(fontsize=20)
# plt.ylim([0.03,0.065])
plt.xlim([-1,27])
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
ax=plt.gca()


#add legend to plot
# plt.legend(bbox_to_anchor=(0.5, -0.18),loc=8,ncol=6,fontsize=20)
x_list = [i for i in range(0,7)] + [i for i in range(9, 16)] + [i for i in range(18, 25)]
x_ticks_list = [f"SID {i}" for i in range(1,8)]
x_ticks_list = x_ticks_list + x_ticks_list + x_ticks_list
plt.xticks(x_list, x_ticks_list, rotation=45, fontsize=14) 

# plt.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(1,2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(16)
plt.yticks(fontsize=16)
plt.savefig(f"./vis/id_nmi.pdf",  bbox_inches='tight')
