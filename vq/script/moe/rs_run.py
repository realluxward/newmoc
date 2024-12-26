# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append('/apdcephfs_cq10/share_1362653/taolinzhang/rs/vector-quantize-pytorch')

from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from IPython import embed
import h5py
from math import ceil
import os
from tqdm import tqdm
import wandb

lr = 3e-4
train_iter = 100000
num_codes = 5000

seed = 1234
num_quantizers = 3
expert_nums = int(sys.argv[1])
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = f"cuda:{sys.argv[2]}"
use_wandb = False

class VQExpert(nn.Module):
    def __init__(self, in_feat, hidden, out_feat):
        super().__init__()
        self.vq = VectorQuantize(dim=hidden, codebook_dim=8, codebook_size = num_codes, kmeans_init=True, commitment_weight=0)
        self.down = nn.Linear(in_feat, hidden)
        self.up = nn.Linear(hidden, out_feat)

        return

    def forward(self, x):
        x = self.down(x)
        x, indices, commit_loss = self.vq(x)
        x = self.up(x)

        return x.clamp(-1, 1), indices, commit_loss
    

class SimpleVQVAE(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.down = nn.Sequential(
            *[
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
            ]
        )
        self.vq = nn.ModuleList(
            # [VQExpert(in_feat=128+i*64, hidden=128, out_feat=128) for i in range(expert_nums)]
            [VQExpert(in_feat=64, hidden=32, out_feat=64) for i in range(expert_nums)]
            # [ResidualVQ(dim=64, codebook_dim=32, num_quantizers = num_quantizers, codebook_size = num_codes, kmeans_init=True, commitment_weight=0) for _ in range(expert_nums)]
        )
        self.up = nn.Sequential(
            *[
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ]
        )
        return

    def forward(self, x):
        # down
        x = self.down(x)
        
        # vq
        # x = torch.cat([rs_x, x], dim=1)
        moe_x = []
        indices = []
        commit_loss = 0
        for i in range(expert_nums):
            # rs_x_i = torch.repeat_interleave(rs_x, i, dim=1)
            # x_i = torch.cat([x, rs_x_i], dim=1)
            x_i = torch.cat([x], dim=1)
            x_i, indices_i, commit_loss_i = self.vq[i](x_i)
            moe_x.append(x_i)
            indices.append(indices_i)
            commit_loss += commit_loss_i
        moe_x = torch.stack(moe_x, dim=0)
        x = torch.mean(moe_x, dim=0)
        # bs, expert_num
        indices = torch.stack(indices, dim=1)

        x = self.up(x)

        return x.clamp(-1, 1), indices, commit_loss


def train(model, item_feat, rs_item_feat, train_iterations=1000, alpha=10):
    pbar = trange(train_iterations)
    bs = 5000
    step = ceil(item_feat.shape[0]/bs)
    for idx in pbar:
        opt.zero_grad()
        x = rs_item_feat[(idx%step)*bs:(idx%step+1)*bs].to(device)
        out, indices, cmt_loss = model(x)
        llm_rec_loss = (out - x).abs().mean()
        rec_loss = llm_rec_loss
        cmt_loss = cmt_loss.mean()
        (rec_loss + alpha * cmt_loss).backward()

        cnt = 0
        for i in range(expert_nums):
            # for j in range(num_quantizers):
            cnt += indices[:, i].unique().numel()
        cnt /= expert_nums

        opt.step()
        pbar.set_description(
            f"llm rec loss: {llm_rec_loss.item():.5f} | "
            + f"cmt loss: {cmt_loss.item():.5f} | "
            + f"active %: {cnt / num_codes * 100:.5f}"
        )
        if use_wandb:
            wandb.log({'llm_rec_loss':llm_rec_loss.item(), 'total_rec loss':rec_loss.item()}, step=idx)
    return

def infer(model, rs_item_feat):
    x = rs_item_feat.to(device)
    out, indices, cmt_loss = model(x)
    return indices

dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"

print("Loading item feat")
# rs_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/lightgcn_item_feat.pt"
# rs_index_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/feat/{split_dataset}/lightgcn_item_list.pt"
# full_rs_feat = torch.load(rs_feat_path, map_location="cpu").detach().clone().numpy()
# full_rs_index = torch.load(rs_index_path, map_location="cpu")
# rs_feat_dict = {}
# for idx in range(len(full_rs_index)):
#     rs_feat_dict[full_rs_index[idx]] = full_rs_feat[idx]

# item_feat_path = f"/apdcephfs_cq10/share_1362653/taolinzhang/llm2vec_data/{dataset}_hdf5/norm_item_feat.h5"
# item_feat_dict = h5py.File(item_feat_path,'r')
# item_feat = []
# rs_item_feat = []
# with open(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/RecBole/dataset/{split_dataset}/{split_dataset}.item", 'r') as read_f:
#     lines = read_f.read().splitlines()
#     for line in tqdm(lines[1:]):
#         item_id = line.split("\t")[0]
#         if item_id not in item_feat_dict.keys() or item_id not in rs_feat_dict.keys():
#             continue
#         item_feat.append(item_feat_dict[item_id][:])
#         rs_item_feat.append(rs_feat_dict[item_id])
# item_feat = np.stack(item_feat)
# item_feat = torch.from_numpy(item_feat)
# rs_item_feat = np.stack(rs_item_feat)
# rs_item_feat = torch.from_numpy(rs_item_feat)
# torch.save(item_feat, f"./feat/{split_dataset}/item_feat_input.pt")
# torch.save(rs_item_feat, f"./feat/{split_dataset}/rs_item_feat_input.pt")
# exit()
item_feat = torch.load(f"./feat/{split_dataset}/item_feat_input.pt")
rs_item_feat = torch.load(f"./feat/{split_dataset}/rs_item_feat_input.pt")

print(f"training with {expert_nums} experts and {num_codes} clusters")
if use_wandb:
    wandb.init()
torch.random.manual_seed(seed)
model = SimpleVQVAE(codebook_size=num_codes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, item_feat, rs_item_feat, train_iterations=train_iter)
model.eval()
indices = infer(model, rs_item_feat)
os.makedirs(f"ckpt/{dataset}/",exist_ok=True)
print(f"save to ./ckpt/{split_dataset}/rs_{num_codes}_{expert_nums}_index.pt")
torch.save(indices, f"./ckpt/{split_dataset}/rs_{num_codes}_{expert_nums}_index.pt")

