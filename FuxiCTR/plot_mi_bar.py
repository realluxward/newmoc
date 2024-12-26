import torch
from matplotlib import pyplot as plt
from IPython import embed
from collections import defaultdict
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
width = 0.8
cmap = plt.get_cmap('tab20').colors
div=0.07015

nomix_data = [0.04032, 0.04606,  0.06055, 0.0703]
base_data = [0.04117,  0.04862, 0.06307, 0.07057]

nomix_data = np.array(nomix_data)
base_data = np.array(base_data)

nomix_data = nomix_data[:-1]
base_data = base_data[:-1]

color_list = {
    "nomix": "#00cbbf",
    "base": "#ff704c"
}

if plot_bar:
    bar_data = defaultdict(list)
    
    x = np.arange(3)

    plt.figure(figsize=(8, 8))

    plt.bar(x, base_data, width=width, label="mixing improvement", color=color_list["base"])

    plt.bar(x, nomix_data, width=width, label="w/o mixing", color=color_list["nomix"])

    
    for cur_x, cur_y in zip(x, base_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.4f}", ha='center', va='bottom', fontsize=20, color='black')

    for cur_x, cur_y in zip(x, nomix_data):
        plt.text(cur_x, cur_y-0.0003, f"{cur_y:.4f}", ha='center', va='top', fontsize=20, color='white')

    plt.xticks(x,['item feat', 'flatten', 'final'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0.03,0.065])
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    ax=plt.gca()

    #specify order of items in legend
    order = [1,0]

    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(0.5, -0.18),loc=8,ncol=4,fontsize=20) 

    # plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f"./vis/mix.pdf",  bbox_inches='tight')
