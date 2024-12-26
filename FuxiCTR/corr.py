import torch
import sys
from IPython import embed
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
cat_num = int(sys.argv[1])
prefix_list = [f"nomix_toys-split-moe-{cat_num}", f"toys-split-rq-{cat_num}", f"toys-split-me-{cat_num}"]
name_list = ["moe","rq","me"]
dump_corr = False
plot_corr = True

feat_dict = {}
def cal_pccs(x,y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    xmean = x.mean(dim=0,keepdim=True)
    ymean = y.mean(dim=0,keepdim=True)
    return torch.sum((x-xmean)*(y-ymean))/torch.sqrt(torch.sum((x-xmean)**2)*torch.sum((y-ymean)**2))

if dump_corr:
    for prefix, name in zip(prefix_list, name_list):
        print(f"----{prefix}---")
        feat_dict[prefix] = torch.load(f"model_zoo/DCNv2/feat/emb/sample_{prefix}_all_cat_64.pt", map_location="cpu")
        bs = feat_dict[prefix].shape[0]
        feat_dict[prefix] = feat_dict[prefix].reshape(bs,-1,64)

        corr = torch.zeros((cat_num,cat_num))
        for i in range(cat_num):
            for j in range(cat_num):
                feat_i = feat_dict[prefix][:, i]
                feat_j = feat_dict[prefix][:, j]
                corr[i,j] = cal_pccs(feat_i, feat_j)
        print(torch.round(corr, 2))
        torch.save(corr, f"./model_zoo/DCNv2/feat/emb/corr_{name}_{cat_num}.pt")

if plot_corr:
    for name in name_list:
        plt.figure(figsize=(8, 8))
        data = torch.load(f"./model_zoo/DCNv2/feat/emb/corr_{name}_{cat_num}.pt")
        data = data.detach().numpy()
        plt.imshow(data, cmap="Blues")
        # if name == "me":
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.clim([0,0.025])
        plt.yticks(np.arange(7), [f"{i}" for i in range(1,8)],fontsize=20)
        plt.xticks(np.arange(7), [f"{i}" for i in range(1,8)],fontsize=20)

        handles, labels = plt.gca().get_legend_handles_labels()
        ax=plt.gca()
        # plt.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(f"./vis/corr_{name}.pdf")