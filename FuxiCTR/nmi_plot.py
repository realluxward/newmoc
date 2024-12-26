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
plot_bar = False
plot_line = True
plot_scale_line = False
hidden = 64
width = 1
cmap = plt.get_cmap('tab20').colors
div=1

if plot_bar:
    bar_data = defaultdict(list)

    color_list = {
        "RQ-VAE": "#35c40e",
        "ME": "#00cbbf",
        "MOC": "#ff704c",
    }
    plot_data = {
        "ME": np.array([0.01251, 0.01245, 0.01172, 0.0124, 0.01242, 0.01326, 0.0126, 0.01356]),
        "RQ-VAE": np.array([0.01567, 0.00308, 0.00135, 0.0011, 0.00116, 0.00098, 0.00116, 0.01473]),
        "MOC": np.array([0.00946, 0.01167, 0.01053, 0.00925, 0.00713, 0.01025, 0.00789, 0.01702]),
    }
    
    total_data = np.stack([plot_data["ME"], plot_data["RQ-VAE"], plot_data["MOC"]], axis=0)
    plot_keys = [f"SID {i}" for i in range(1,8)] + ["Flatten"]

    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap('Blues')
    scale = 0.006
    norm = Normalize(vmin=total_data.min()-scale, vmax=total_data.max()+scale)


    for i in range(len(plot_keys)):
        x = np.arange(i, i+31.5, 10.5)
        color = norm(total_data[:, i])
        plt.bar(x, total_data[:, i], width=width, label=plot_keys[i], color=cmap(color))
        if i == len(plot_keys)-1:
            # plt.plot(x, total_data[:, i], marker = 'o', linewidth=1)
            for j in range(3):
                if j == 0:
                    add = 0
                else:
                    add = 0
                plt.text(x[j]+add, total_data[:, i][j]+0.0003, f"{total_data[:, i][j]:.4f}", ha='center', va='bottom', fontsize=14, color='black')
    
    plt.text(4, -0.003, f"ME", ha='center', va='bottom', fontsize=20, color='black')

    plt.text(14, -0.003, f"RQ-VAE", ha='center', va='bottom', fontsize=20, color='black')

    plt.text(25, -0.003, f"MOC", ha='center', va='bottom', fontsize=20, color='black')


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
    plt.xlim([-1,30])
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    ax=plt.gca()


    #add legend to plot
    # plt.legend(bbox_to_anchor=(0.5, -0.18),loc=8,ncol=6,fontsize=20)
    x_list = [i for i in range(0,8)] + [i+0.5 for i in range(10, 18)] + [i for i in range(21, 29)]
    x_ticks_list = [f"SID {i}" for i in range(1,8)] + ['Flatten']
    x_ticks_list = x_ticks_list + x_ticks_list + x_ticks_list
    plt.xticks(x_list, x_ticks_list, rotation=45, fontsize=14) 

    # plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.ticklabel_format(style='sci', scilimits=(1,2), axis='y')
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.yticks(fontsize=16)
    plt.savefig(f"./vis/nmi.pdf",  bbox_inches='tight')

if plot_line:
    cmap = plt.get_cmap('viridis').colors
    color_list = {
        f"ME": "#482976",
        f"RQ-VAE": "#30688D",
        f"MOC": "#32B67B",
    }
    plot_data = {
        "ME": np.array([0.0, 0.00828, 0.00878, 0.0091, 0.01066, 0.00989, 0.0113, 0.01258, 0.01179, 0.01268, 0.01356, 0.01374, 0.01421, 0.01451, 0.0151, 0.01535, 0.0158, 0.01559, 0.01652, 0.01666, 0.01651]),
        "RQ-VAE": np.array([0.0, 0.01323, 0.01322, 0.01384, 0.01291, 0.01336, 0.01432, 0.01361, 0.01456, 0.0145, 0.01473, 0.01474, 0.0151, 0.01534, 0.01442, 0.01507, 0.01444, 0.01515, 0.01567, 0.01505, 0.01572]),
        "MOC": np.array([0.0, 0.0157, 0.01575, 0.01607, 0.01598, 0.01641, 0.01704, 0.01734, 0.01741, 0.01726, 0.01702, 0.01756, 0.01852, 0.01772, 0.01815, 0.01884, 0.01887, 0.01866, 0.01875, 0.01894, 0.01897]) 
    }
    plt.figure(figsize=(6, 6))
    for idx, k in enumerate(plot_data.keys()):
        x = [10*i for i in range(21)]
        y = np.array(plot_data[k])/div
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
    plt.savefig("./vis/nmi_line.pdf",bbox_inches='tight')
