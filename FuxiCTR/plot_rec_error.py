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
width = 0.6
cmap = plt.get_cmap('tab20').colors
div=0.07015

llm_data = [0.38]
idx_1_data = [99.41]
idx_2_data = [91.57]
idx_3_data = [68.64]

llm_data = np.array(llm_data)
idx_1_data = np.array(idx_1_data)
idx_2_data = np.array(idx_2_data)
idx_3_data = np.array(idx_3_data)

# llm_data = np.round(llm_data, 2)
# rs_data = np.round(rs_data, 2)
# id_data = np.round(id_data,2)
# colors = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#1B9E8A', '#32B67B', '#6CCC5F', '#B4DD3D', '#FDE73A']

color_list = {
    "idx_1": "#482976",
    "idx_2": "#30688D",
    "idx_3": "#1B9E8A",
    "llm": "#6CCC5F"
}

if plot_bar:
    bar_data = defaultdict(list)
    
    llm_x = np.array([4])
    idx_1_x = np.array([1])
    idx_2_x = np.array([2])
    idx_3_x = np.array([3])

    plt.figure(figsize=(6, 6))

    plt.bar(1, idx_1_data, width=width, color=color_list["idx_1"])
    plt.bar(2, idx_2_data, width=width, color=color_list["idx_2"])
    plt.bar(3, idx_3_data, width=width, color=color_list["idx_3"])
    # plt.bar(4, llm_data, width=width, color=color_list["llm"])

    
    # for cur_x, cur_y in zip(llm_x, llm_data):
    #     plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')

    for cur_x, cur_y in zip(idx_1_x, idx_1_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')
    
    for cur_x, cur_y in zip(idx_2_x, idx_2_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')
    
    for cur_x, cur_y in zip(idx_3_x, idx_3_data):
        plt.text(cur_x, cur_y+0.0003, f"{cur_y:.2f}", ha='center', va='bottom', fontsize=20, color='black')
    

    plt.xticks([1,2,3],["1x", "2x", "3x"], fontsize=20)
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
    # plt.yscale('log')
    plt.xlabel('Semantic ID', fontsize=20)
    plt.ylabel('Reconstruction Error', fontsize=20)
    plt.ylim([50, 100])
    # ax.ticklabel_format(style='sci', scilimits=(-1,6), axis='y')
    plt.savefig(f"./vis/rec_error.pdf",  bbox_inches='tight')
