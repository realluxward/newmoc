import torch
import h5py
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
from IPython import embed
from math import ceil

dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"
train_iterations = 5
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False
if preprocess:
    rs_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/lightgcn_item_feat.pt"
    rs_index_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/lightgcn_item_list.pt"
    full_rs_feat = torch.load(rs_feat_path, map_location="cpu").detach().clone().numpy()
    full_rs_index = torch.load(rs_index_path, map_location="cpu")
    rs_feat_dict = {}
    for idx in range(len(full_rs_index)):
        rs_feat_dict[full_rs_index[idx]] = full_rs_feat[idx]

    item_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/llm2vec_data/{dataset}_hdf5/norm_item_feat.h5"
    item_feat_dict = h5py.File(item_feat_path,'r')
    item_feat = []
    rs_item_feat = []
    with open(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset/{split_dataset}/{split_dataset}.item", 'r') as read_f:
        lines = read_f.read().splitlines()
        for line in tqdm(lines[1:]):
            item_id = line.split("\t")[0]
            if item_id not in item_feat_dict.keys() or item_id not in rs_feat_dict.keys():
                continue
            item_feat.append(item_feat_dict[item_id][:])
            rs_item_feat.append(rs_feat_dict[item_id])
    item_feat = np.stack(item_feat)
    item_feat = torch.from_numpy(item_feat)
    rs_item_feat = np.stack(rs_item_feat)
    rs_item_feat = torch.from_numpy(rs_item_feat)
    torch.save(item_feat, f"./feat/{split_dataset}/item_feat_input.pt")
    torch.save(rs_item_feat, f"./feat/{split_dataset}/rs_item_feat_input.pt")

    rs_item_feat_dict = {}
    item_feat_dict = {}
    sample_item_feat = []
    sample_rs_item_feat = []
    idx = 0
    with open(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset/{split_dataset}/{split_dataset}.item", 'r') as read_f:
        lines = read_f.read().splitlines()
        for line in tqdm(lines[1:]):
            item_id = line.split("\t")[0]
            item_feat_dict[item_id] = item_feat[idx]
            rs_item_feat_dict[item_id] = rs_item_feat[idx]
            idx += 1

    for postfix in ["train", "valid", "test"]:
        with open(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset/{split_dataset}/{split_dataset}.{postfix}.inter", 'r') as read_f:
            lines = read_f.read().splitlines()
            for line in tqdm(lines[1:]):
                item_id = line.split("\t")[1]
                sample_item_feat.append(item_feat_dict[item_id])
                sample_rs_item_feat.append(rs_item_feat_dict[item_id])

    print("Stacking...")
    sample_item_feat = torch.stack(sample_item_feat).cpu()
    sample_rs_item_feat = torch.stack(sample_rs_item_feat).cpu()

    print("Saving...")
    torch.save(sample_item_feat, f"./feat/{split_dataset}/sample_item_feat_input.pt")
    torch.save(sample_rs_item_feat, f"./feat/{split_dataset}/sample_rs_item_feat_input.pt")
    exit()

print("Loading data...")
sample_item_feat = torch.load(f"./feat/{split_dataset}/sample_item_feat_input.pt", map_location="cpu")
sample_rs_item_feat = torch.load(f"./feat/{split_dataset}/sample_rs_item_feat_input.pt", map_location="cpu")
# sample_item_feat = torch.load(f"./feat/{split_dataset}/item_feat_input.pt", map_location="cpu")
# sample_rs_item_feat = torch.load(f"./feat/{split_dataset}/rs_item_feat_input.pt", map_location="cpu")

print("SVD...")
if shuffle:
    perm = torch.randperm(sample_item_feat.shape[0])
    sample_item_feat = sample_item_feat[perm]
    sample_rs_item_feat = sample_rs_item_feat[perm]
step = ceil(sample_item_feat.shape[0]/bs)
pbar = trange(train_iterations)
total_s = []
total_rs_s = []
for idx in pbar:
    x = sample_item_feat[(idx%step)*bs:(idx%step+1)*bs].to(device)
    rs_x = sample_rs_item_feat[(idx%step)*bs:(idx%step+1)*bs].to(device)
    u,s,v = torch.linalg.svd(x, full_matrices=False)
    total_s.append(s)
    rs_u, rs_s, rs_v = torch.linalg.svd(rs_x, full_matrices=False)
    total_rs_s.append(rs_s)
total_s = torch.stack(total_s)
total_rs_s = torch.stack(total_rs_s)
avg_s = total_s.mean(dim=0)
avg_rs_s = total_rs_s.mean(dim=0)
print(avg_s.shape)
print(avg_rs_s.shape)
ia_s = avg_s.sum()/avg_s.max()
ia_rs_s = avg_rs_s.sum()/avg_rs_s.max()
print(f"{ia_s}, {ia_rs_s}")
torch.save(avg_s, f"./feat/{split_dataset}/avg_s.pt")
torch.save(avg_rs_s, f"./feat/{split_dataset}/avg_rs_s.pt")