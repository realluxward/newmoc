import torch
import h5py
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
from IPython import embed
from math import ceil
import sys

dataset = f"amazon-toys-games-filter"
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
hidden = 64
concat = False
extend = False
# prefix = "nomix"
cat_num = int(sys.argv[1])
# prefix_list = [f"base_toys-split-moe-{cat_num}", f"nomix_toys-split-moe-{cat_num}", f"toys-split-rq-{cat_num}", f"toys-split-me-{cat_num}"]
prefix_list = [f"nomix_toys-split-moe-{cat_num}", f"toys-split-rq-{cat_num}", f"toys-split-me-{cat_num}"]

# svd_keys = ['item_id', 'user_id','brand', 'categories','sales_type']
# svd_keys = ['item_id', 'user_id','brand', 'categories','sales_type','moe_cid_1']
# svd_keys = ['moe2_item_id', 'moe2_user_id', 'moe2_brand', 'moe2_categories', 'moe2_sales_type', 'moe2_moe_cid_1', 'moe2_moe_cid_2']
# svd_keys = ["flat"]
svd_keys = ["all_cat"]

if concat:
    # cat_keys = ['item_id', 'moe_cid_1','moe_cid_2']
    for prefix in prefix_list:
        cat_keys = [f'{prefix}_moe_cid_{i}' for i in range(1,cat_num+1)]
        cat_feat_dict = {}
        feat_dict = {}
        for i in range(len(cat_keys)):
            feat_dict[cat_keys[i]] = torch.load(f"./feat/emb/sample_{cat_keys[i]}_64.pt", map_location="cpu")
        # for i in range(3):
        #     for j in range(i+1,3):
        #         cat_feat_dict[f"<{cat_keys[i]},{cat_keys[j]}>"] = torch.cat([feat_dict[cat_keys[i]], feat_dict[cat_keys[j]]], dim=1)
        
        # cat_feat_dict[f"<{cat_keys[0]},{cat_keys[1]},{cat_keys[2]}>"] = torch.cat([feat_dict[cat_keys[0]], feat_dict[cat_keys[1]], feat_dict[cat_keys[2]]], dim=1)
        cat_feat_dict[f"all_cat"] = torch.cat([
            feat_dict[k] for k in cat_keys
        ], dim=1)
        for k in cat_feat_dict.keys():
            torch.save(cat_feat_dict[k],f"./feat/emb/sample_{prefix}_{k}_{hidden}.pt")
    svd_keys += list(cat_feat_dict.keys())


if extend:
    # item_feat = torch.load(f"./feat/emb/sample_item_id_64.pt", map_location="cpu")
    # extend_feat = torch.load(f"./feat/emb/sample_extend_moe_cid_1_128_64.pt", map_location="cpu")
    # concat_extend_feat = torch.cat([item_feat, extend_feat], dim=1)
    # torch.save(concat_extend_feat,f"./feat/emb/sample_<item_id,extend_moe_cid_1_128>_64.pt")
    extend_keys = ["extend_moe_cid_1_192"]
    svd_keys += extend_keys
    
for prefix in prefix_list:
    for k in svd_keys:
        print("Loading data...")
        print(f"{prefix}_{k}")
        sample_item_feat = torch.load(f"./feat/emb/sample_{prefix}_{k}_{hidden}.pt", map_location="cpu")
        # sample_moe_feat = torch.load(f"./feat/sample_moe_1_emb.pt", map_location="cpu")

        print("SVD...")
        if shuffle:
            perm = torch.randperm(sample_item_feat.shape[0])
            sample_item_feat = sample_item_feat[perm]
            # sample_moe_feat = sample_moe_feat[perm]
        step = ceil(sample_item_feat.shape[0]/bs)
        pbar = trange(train_iterations)
        total_s = []
        for idx in pbar:
            x = sample_item_feat[(idx%step)*bs:(idx%step+1)*bs].to(device)
            # moe_x = sample_moe_feat[(idx%step)*bs:(idx%step+1)*bs].to(device)
            u,s,v = torch.linalg.svd(x, full_matrices=False)
            total_s.append(s)
            # rs_u, rs_s, rs_v = torch.linalg.svd(moe_x, full_matrices=False)
            # total_rs_s.append(rs_s)
        total_s = torch.stack(total_s)
        avg_s = total_s.mean(dim=0)
        print(avg_s.shape)
        ia_s = avg_s.sum()/avg_s.max()
        print(f"{avg_s.mean()}")
        torch.save(avg_s, f"./feat/sigma/sigma_{prefix}_{k}_{hidden}.pt")