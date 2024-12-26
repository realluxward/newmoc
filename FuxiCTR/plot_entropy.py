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

llm_data = [28517.305]
rs_data = [429.214]
id_data = [7.925]

llm_data = np.array(llm_data)
rs_data = np.array(rs_data)
id_data = np.array(id_data)

# llm_data = np.round(llm_data, 2)
# rs_data = np.round(rs_data, 2)
# id_data = np.round(id_data,2)

color_list = {
    "id": "#35c40e",
    "llm": "#ff704c",
    "rs": "#00cbbf"
}

if plot_bar:
    bar_data = defaultdict(list)
    
    id_x = np.array([2.5])
    rs_x = np.array([4])
    llm_x = np.array([1])

    plt.figure(figsize=(8, 8))

    plt.bar(llm_x, llm_data, width=width, label="LLM Embedding", color=color_list["llm"])
    plt.bar(rs_x, rs_data, width=width, label="RS Embedding", color=color_list["rs"])
    plt.bar(id_x, id_data, width=width, label="Semantic ID", color=color_list["id"])

    
    for cur_x, cur_y in zip(llm_x, llm_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')

    for cur_x, cur_y in zip(rs_x, rs_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')
    
    for cur_x, cur_y in zip(id_x, id_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')

    plt.xticks([1,2.5,4],["LLM Embedding", "Semantic ID", "RS Embedding"], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim([0.03,0.065])
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    ax=plt.gca()

    # #specify order of items in legend
    # order = [1,0]

    # #add legend to plot
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(0.5, -0.18),loc=8,ncol=4,fontsize=20) 

    # plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yscale('log')
    plt.ylim([1, 30000])
    # ax.ticklabel_format(style='sci', scilimits=(-1,6), axis='y')
    plt.savefig(f"./vis/entropy.pdf",  bbox_inches='tight')
