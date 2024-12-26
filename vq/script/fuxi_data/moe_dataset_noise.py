import torch
from IPython import embed
import torch
import os
import sys
from tqdm import tqdm
import h5py
import sys
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

n_cluster = 1000
dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"

index_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/vector-quantize-pytorch/ckpt/{split_dataset}/moe_{n_cluster}_{sys.argv[1]}_index_noise.pt"
print(f"Load from {index_path}")
index = torch.load(index_path)

expert_nums = index.shape[1]

rs_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/dcnv2_item_feat.pt"
rs_index_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/dcnv2_item_list.pt"
full_rs_feat = torch.load(rs_feat_path, map_location="cpu").detach().clone().numpy()
full_rs_index = torch.load(rs_index_path, map_location="cpu")
rs_feat_dict = {}
for idx in range(len(full_rs_index)):
    rs_feat_dict[full_rs_index[idx]] = full_rs_feat[idx]

item_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/llm2vec_data/{dataset}_hdf5/norm_item_feat.h5"
item_feat_dict = h5py.File(item_feat_path,'r')
item_cluster_id_dict = {}
index_cnt = 0
with open(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset/{split_dataset}/{split_dataset}.item", 'r') as read_f:
    lines = read_f.read().splitlines()
    for line in tqdm(lines[1:]):
        item_id = line.split("\t")[0]
        if item_id not in item_feat_dict.keys() or item_id not in rs_feat_dict.keys():
            continue
        item_cluster_id_dict[item_id] = index[index_cnt]
        index_cnt += 1
assert(len(item_cluster_id_dict.keys())==index_cnt)



fuxi = "/apdcephfs_cq10/share_1362653/taolinzhang/rs/FuxiCTR"
dataset_path = f"{fuxi}/data"
root = f"{dataset_path}/{split_dataset}-moe-{expert_nums}-noise"
bole = "/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset"
refer = f"{bole}/{split_dataset}/{split_dataset}"
if not os.path.exists(root):
    os.makedirs(root)

print("Encode item id...")
item_data = defaultdict(dict)
feat_keys = ["item_id", "sales_type", "brand", "categories"]
total_data = defaultdict(list)
with open(f"{refer}.item", 'r') as read_f:
    lines = read_f.read().splitlines()
    for line in tqdm(lines[1:]):
        item_id,title,price,sales_type,sales_rank, brand, categories = line.split("\t")
        item_data[item_id] = {"item_id": item_id, "sales_type": sales_type, "brand": brand, "categories": categories}
        for feat_key in feat_keys:
            total_data[feat_key].append(item_data[item_id][feat_key])
encode_item_data = defaultdict(dict)
for feat_key in feat_keys:
    lbe = LabelEncoder()
    lbe.fit(total_data[feat_key])
    total_data[feat_key] = lbe.transform(total_data[feat_key])
    for idx,item_id in enumerate(item_data.keys()):
        encode_item_data[item_id][feat_key] = total_data[feat_key][idx]

print("Encode user id...")
total_user_id = []
for postfix in ["train", "valid","test"]:
    with open(f"{refer}.{postfix}.inter", 'r') as read_f:
        inters = read_f.read().splitlines()
        inters = inters[1:]
        for inter in tqdm(inters):
            user_id,item_id,rating,timestamp = inter.split("\t")
            total_user_id.append(user_id)
user_lbe = LabelEncoder()
user_lbe.fit(total_user_id)
encode_user_data = {}
for idx in range(user_lbe.classes_.shape[0]):
    encode_user_data[user_lbe.classes_[idx]] = idx



print("Write data...")
for postfix in ["train", "valid", "test"]:
    with open(f"{root}/{postfix}.csv", 'w') as write_f, open(f"{refer}.{postfix}.inter", 'r') as read_f:
        inters = read_f.read().splitlines()
        write_f.write(f"user_id,item_id,label,sales_type,brand,categories")
        for i in range(1,expert_nums+1):
            write_f.write(f",moe_cid_{i}")
        write_f.write("\n")
        inters = inters[1:]
        for inter in tqdm(inters):
            user_id,item_id,rating,timestamp = inter.split("\t")
            label = int(float(rating)>3)
            if item_id in item_cluster_id_dict.keys():
                encode_user_id = encode_user_data[user_id]
                encode_item_id = encode_item_data[item_id]["item_id"]
                sales_type, brand, categories =  encode_item_data[item_id]["sales_type"], encode_item_data[item_id]["brand"], encode_item_data[item_id]["categories"]
                moe_cid = item_cluster_id_dict[item_id]
                write_f.write(f"{encode_user_id},{encode_item_id},{label},{sales_type},{brand},{categories}")
                for i in range(expert_nums):
                    # write_f.write(f",{moe_cid[i]}")
                    write_f.write(f",{moe_cid[0]}")
                write_f.write("\n")
