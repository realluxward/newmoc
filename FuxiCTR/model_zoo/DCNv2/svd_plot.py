import torch
from matplotlib import pyplot as plt
from IPython import embed
import sys
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
hidden = 64
dim = sys.argv[1]

plt.figure()
# plot_keys = ['item_id','moe_cid_1','moe_cid_2']
if dim=="1":
    # plot_keys = ['item_id', 'user_id','brand', 'categories','sales_type','moe_cid_1']
    # plot_keys = ['item_id', 'moe_cid_1']
    plot_keys = ['moe2_item_id', 'moe2_user_id', 'moe2_brand', 'moe2_categories', 'moe2_sales_type', 'moe2_moe_cid_1', 'moe2_moe_cid_2']
    # plot_keys = ['moe2_item_id', 'moe2_user_id', 'moe2_moe_cid_1', 'moe2_moe_cid_2']
elif dim in ["2","3","5","6","7"]:
    plot_keys = [f"{prefix}_all_cat" for prefix in [f"nomix_toys-split-moe-{dim}", f"toys-split-rq-{dim}", f"toys-split-me-{dim}"]]
    color_list = ["#ff704c","#20A586", "#470F62"]
    name_list = ["MOC", "RQ-VAE", "ME"]
elif dim in ["mix"]:
    plot_keys = [f"{prefix}_all_cat" for prefix in [f"nomix_toys-split-moe-{7}", f"base_toys-split-moe-{7}"]]
    color_list = ["#482976","#32B67B"]
    name_list = ["MOC w/o Fusion", "MOC w/ Fusion"]
elif dim=="3":
    # plot_keys = ["<item_id,extend_moe_cid_1_128>", "<item_id,moe_cid_1,moe_cid_2>"]
    # plot_keys = ["extend_moe_cid_1_192", "<moe_cid_1,moe_cid_2,moe_cid_3>"]
    plot_keys = [f"cat_moe3"] + [f"cat_scale3"]
elif dim=="4":
    plot_keys = [f"cat_moe4"] + [f"cat_scale4"]
elif dim=="0":
    plot_keys = []
feat_dict = {}
for idx, k in enumerate(plot_keys):
    avg_s=torch.load(f"./feat/sigma/sigma_{k}_{hidden}.pt", map_location="cpu")
    avg_s = avg_s.detach().cpu().numpy()
    feat_dict[k] = avg_s
    # avg_s = avg_s[:30]
    # s = avg_s/avg_s.max()
    # s = avg_s/avg_s.sum()
    s = avg_s
    # print(s.sum()/s.max())
    print(s.mean())
    # plt.ylim([1e-2,30])
    # plt.plot(s, label = k)
    # plt.yscale('log')
    plt.plot(s, c=color_list[idx], label=name_list[idx])
plt.xticks(fontsize=20)
# plt.xlim([-0.5,25])
plt.yticks(fontsize=20)
# plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
ax=plt.gca()

#specify order of items in legend
# order = [2, 1,0]

# #add legend to plot
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=20) 
plt.legend(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f"./vis/sigma_{dim}.pdf", bbox_inches='tight')
# print((feat_dict[plot_keys[0]]>=feat_dict[plot_keys[1]]).sum())
# print((feat_dict[plot_keys[1]]>=feat_dict[plot_keys[2]]).sum())