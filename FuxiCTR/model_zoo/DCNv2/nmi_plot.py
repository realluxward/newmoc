import torch
from matplotlib import pyplot as plt
from IPython import embed
import sys
import numpy as np
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
plot_bar = True
plot_line = False
hidden = 64
width = 0.1
cmap = plt.get_cmap('tab20').colors
div=0.07015

if plot_bar:
    bar_data = {
        'item_id': [0.04595, 0.04564, 0.04515],
        'cid_1': [0.01545, 0.01392],
        'cid_2': [0.01574],
        '<item_id, cid_1>': [0.04737, 0.04829],
        '<item_id, cid_2>': [0.04818],
        '<item_id, cid_1, cid_2>': [0.04933],
        'flatten': [0.05608, 0.05771, 0.0582]
    }
    keys = ['item_id', 'cid_1', 'cid_2', '<item_id, cid_1>', '<item_id, cid_2>', '<item_id, cid_1, cid_2>', 'flatten']

    start_idx = np.array([0.0, 0.4, 1.0])

    plt.figure(figsize=(8, 8))
    for idx, k in enumerate(keys):
        count = len(bar_data[k])
        x = start_idx[3-count:]
        y = np.array(bar_data[k])/div
        plt.bar(x, y, width=width, label=k, color=cmap[idx])
        for cur_x, cur_y in zip(x, y):
            plt.text(cur_x, cur_y, f"{cur_y:.3f}", ha='center', va='bottom', fontsize=8, color=cmap[idx])
        start_idx[3-count:] += width
    plt.xticks([0.05, 0.55, 1.3],['Base', 'moe_cid_1', 'moe_cid_2'])
    plt.yticks(fontsize=15)
    plt.ylim([0.1,1])
    plt.legend(bbox_to_anchor=(0.5, -0.14),loc=8,ncol=4,fontsize=10)
    plt.savefig(f"./vis/nmi.pdf")

if plot_line:
    plot_data = {
        'base': [0.05608, 0.06471, 0.0688],
        'moe_cid_1':[0.05771,0.06598, 0.06985 ],
        'moe_cid_2': [0.0582, 0.06616, 0.07015]
    }
    marker_list = ['o', '*', '^']
    plt.figure(figsize=(8, 8))
    for idx, k in enumerate(plot_data.keys()):
        x = np.arange(3)
        y = np.array(plot_data[k])/div
        plt.plot(x, y, label=k, color=cmap[idx], linewidth=2, marker=marker_list[idx], markersize=8)
    plt.grid()
    plt.xticks([0, 1, 2],['flatten', 'final', 'pred'])
    plt.legend(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("./vis/nmi_line.pdf")