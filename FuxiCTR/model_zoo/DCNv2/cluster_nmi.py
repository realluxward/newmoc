import torch
from IPython import embed
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from  sklearn.metrics import roc_auc_score,accuracy_score, normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
import sys

pred_norm = False
use_moe_feat = True
use_cat_feat = False
use_layer_feat = True
moe_num = 0
cat_num = 0
# n_clusters_list = [i for i in range(1,10)]+[10*i for i in range(1,11)]
# n_clusters_list = [i for i in range(1,10)]
n_clusters_list = [10, 20, 30,40,50,60,70,80,90,100] + [110, 120, 130,140,150,160,170,180,190,200]

num = int(sys.argv[1])
# prefix = f"toys-split"
# prefix = f"base_toys-split-rq-{num}"
# prefix = f"base_toys-split-moe-{num}"
# prefix = f"toys-split-rq-{num}"
# prefix = f"nomix_toys-split-moe-{num}"
# prefix = f"toys-split-me-{num}"
# prefix_list = [f"toys-split"]
prefix_list = [f"nomix_split-me-{num}"]
# prefix_list = [f"nomix_split-moe-{num}"]
# prefix_list = [f"nomix_split-rq-{num}"]
# prefix_list = [f"nomix_split-moe-{num}", f"nomix_split-rq-{num}", f"nomix_split-me-{num}"]
# prefix_list = [f"base_toys-split-moe-{num}", f"base_toys-split-rq-{num}", f"toys-split-me-{num}"]
# prefix_list = [f"nomix_split-moe-{i}" for i in range(1,8)]
# prefix_list = [f"nomix_split-rq-{i}" for i in range(1,8)]
# prefix_list = [f"nomix_split-me-{i}" for i in range(1,8)]

stack_results = []
for prefix in prefix_list:
    print(prefix)
    label = torch.load(f"./feat/id/sample_{prefix}_label_64.pt", map_location="cpu").view(-1)

    pred = torch.load(f"./feat/emb/sample_{prefix}_pred_64.pt", map_location="cpu")
    pred = pred.detach().numpy()

    item_id_feat = torch.load(f"./feat/emb/sample_{prefix}_item_id_64.pt", map_location="cpu")
    item_id_feat = item_id_feat.detach().numpy()
    if use_moe_feat:
        moe_feat_dict = {}
        for i in range(1, moe_num+1):
            moe_feat_dict[f"moe_cid_{i}_feat"] = torch.load(f"./feat/emb/sample_{prefix}_moe_cid_{i}_64.pt", map_location="cpu")
            moe_feat_dict[f"moe_cid_{i}_feat"] = moe_feat_dict[f"moe_cid_{i}_feat"].detach().numpy()

    if use_cat_feat:
        if cat_num>=1:
            cat_feat_dict = {}
            cat_feat_dict["cat_id_1_feat"] = np.concatenate([item_id_feat, moe_feat_dict[f"moe_cid_1_feat"]], axis=1)
        if cat_num>=2:
            cat_feat_dict["cat_id_2_feat"] = np.concatenate([item_id_feat, moe_feat_dict[f"moe_cid_2_feat"]], axis=1)
            cat_feat_dict["cat_id_1_2_feat"] = np.concatenate([item_id_feat, moe_feat_dict[f"moe_cid_1_feat"], moe_feat_dict[f"moe_cid_2_feat"]], axis=1)

    if use_layer_feat:
        layer_feat_dict = {}
        # layer_feat_name = ["flat"] + [f"cross_{i}" for i in range(3)] + [f"dnn_{i}" for i in range(2)] + ["final", "pred"]
        # layer_feat_name = ["all_cat"] + ["final", "pred"]
        layer_feat_name = ["all_cat"]
        # layer_feat_name = ["moe_cid_1"] 
        # layer_feat_name = ["item_id"]
        for name in layer_feat_name:
            layer_feat_dict[name] = torch.load(f"./feat/emb/sample_{prefix}_{name}_64.pt", map_location="cpu")
            layer_feat_dict[name] = layer_feat_dict[name].detach().numpy()

    mi_list = []
    feat_list = []
    name_list = []

    if use_moe_feat:
        feat_list = [moe_feat_dict[name] for name in moe_feat_dict.keys()] + feat_list
        name_list = list(moe_feat_dict.keys()) + name_list

    if use_cat_feat:
        feat_list += [cat_feat_dict[name] for name in cat_feat_dict.keys()]
        name_list += list(cat_feat_dict.keys())

    if use_layer_feat:
        feat_list += [layer_feat_dict[name] for name in layer_feat_name]
        name_list += layer_feat_name

    # pred_res = (pred>=pred.mean()).astype(int)
    # pred_res = pred_res.reshape(-1)
    # div = mutual_info_score(label, pred_res)
    label_div = 1
    item_id_div = 1
    moe_cid_1_div = 1
    moe_cid_2_div = 1
    all_id_discretization = None

    feat_results = []
    hidden_results = []
    all_results = []
    for n_clusters in n_clusters_list:
        print(f"--- using {n_clusters} clusters ---")
        if pred_norm:
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=1027, n_init=3).fit(pred)
            discretization = np.array(km.labels_)
            label_score = mutual_info_score(label, discretization)
            label_div = label_score
        for feat, name in zip(feat_list, name_list):
            # 使用MiniBatchKMeans进行聚类
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=1027, n_init=3).fit(feat)
            discretization = np.array(km.labels_)
            # print(normalized_mutual_info_score(label, discretization))
            # print(mutual_info_score(label, discretization))
            label_score = mutual_info_score(label, discretization) / label_div
            # item_id_score = mutual_info_score(item_id, discretization) / item_id_div
            # moe_cid_1_score = mutual_info_score(moe_cid_1, discretization) / moe_cid_1_div
            # moe_cid_2_score = mutual_info_score(moe_cid_2, discretization) / moe_cid_2_div
            
            # print(f"{name:20} <-> | label: {np.round(label_score, 5):5}\t| ", end="")
            # print(f"item_id: {np.round(item_id_score, 5):5}\t| ", end="")
            # print(f"moe_cid_1: {np.round(moe_cid_1_score, 5):5}\t| ", end="")
            # print(f"moe_cid_2: {np.round(moe_cid_2_score, 5):5}\t ")
            print(f"{name:20} <-> | label: {np.round(label_score, 5):5}")
            if "feat" in name or 'cat' in name:
                feat_results.append(np.round(label_score, 5))
            if name in ["flat", "final", "pred"]:
                hidden_results.append(np.round(label_score, 5))
            if name in ['all_cat', 'item_cat']:
                stack_results.append(np.round(label_score, 5))
            all_results.append(np.round(label_score, 5))
    print("feat_results:")
    print(feat_results)
    print("hidden_results:")
    print(hidden_results)
    print("all results:")
    print(all_results)
print("stack results:")
print(stack_results)

