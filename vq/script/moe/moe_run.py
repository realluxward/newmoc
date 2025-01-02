# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append('/apdcephfs_cq10/share_1362653/taolinzhang/rs/vector-quantize-pytorch')

from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from IPython import embed
import h5py
from math import ceil
import os
from tqdm import tqdm
import wandb

lr = 3e-4
train_iter = 10000
num_codes = 256 # 类别数目
seed = 1027
num_quantizers = 3 # rq-vae中的中间残差层数
expert_nums = int(sys.argv[1])
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = f"cuda:{sys.argv[2]}"
use_wandb = False
shuffle = True

class VQExpert(nn.Module):
    def __init__(self, in_feat, hidden, out_feat):
        super().__init__()
        self.vq = VectorQuantize(dim=hidden, codebook_dim=32, codebook_size = num_codes, kmeans_init=True, commitment_weight=0)
        self.down = nn.Linear(in_feat, hidden)
        self.up = nn.Linear(hidden, out_feat)
        return

    def forward(self, x, rs_x=None):
        x = self.down(x)
        x, indices, commit_loss = self.vq(x)
        x = self.up(x)
        return x.clamp(-1, 1), indices, commit_loss
    

class SimpleVQVAE(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.llm_down = nn.Sequential(
            *[
                nn.Linear(4096, 512),
                nn.ReLU(),
                # nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                # nn.BatchNorm1d(256),
                nn.Linear(256, 128),
            ]
        )
        self.vq = nn.ModuleList(
            [VQExpert(in_feat=128, hidden=128, out_feat=128) for i in range(expert_nums)]
        )
        self.llm_up = nn.Sequential(
            *[
                nn.Linear(128, 256),
                nn.ReLU(),
                # nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                # nn.BatchNorm1d(512),
                nn.Linear(512, 4096),
            ]
        )
        self.rs_down = nn.Sequential(
            *[
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ]
        )
        self.rs_up = nn.Sequential(
            *[
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ]
        )
        self.rs_emb = nn.Embedding(1000, 128)
        return

    def forward(self, x):
        x = self.llm_down(x)
        moe_x = []
        indices = []
        commit_loss = 0
        for i in range(expert_nums):
            x_i = x
            x_i, indices_i, commit_loss_i = self.vq[i](x_i)
            moe_x.append(x_i)
            indices.append(indices_i)
            commit_loss += commit_loss_i
        moe_x = torch.stack(moe_x, dim=1)
        x = (moe_x).mean(dim=1)
        indices = torch.stack(indices, dim=1)
        x = self.llm_up(x)
        return x.clamp(-1, 1), indices, commit_loss


def train(model, item_feat, rs_item_feat, train_iterations=1000, alpha=1, rs_weight=5):
    pbar = trange(train_iterations)
    bs = 5000
    step = ceil(item_feat.shape[0]/bs)
    for idx in pbar:
        opt.zero_grad()
        if shuffle and idx%step==0:
            perm = torch.randperm(item_feat.shape[0])
            item_feat = item_feat[perm] 
        x = item_feat[(idx%step)*bs:(idx%step+1)*bs]
        if x.shape[0] < bs:
            x = item_feat[-bs:]
        x = x.to(device)
        out, indices, cmt_loss = model(x) #这里全是0
        llm_rec_loss = (out - x).abs().mean()*100
        # rec_loss = rs_rec_loss
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
            # + f"rs rec loss: {rs_rec_loss.item():.5f} | "
            + f"cmt loss: {cmt_loss.item():.5f} | "
            + f"active %: {cnt / num_codes * 100:.5f}"
        )
        if use_wandb:
            wandb.log({'llm_rec_loss':llm_rec_loss.item(), 'rs_rec_loss':rs_rec_loss.item(), 'total_rec loss':rec_loss.item(), 'active':cnt / num_codes * 100}, step=idx)
    return

def infer(model, item_feat, rs_item_feat):
    x = item_feat.to(device)
    out, indices, cmt_loss = model(x)
    return indices

dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"
# dataset = f"amazon-beauty-filter"
# split_dataset = f"beauty-split"
# dataset = f"amazon-sports-outdoors-filter"
# split_dataset = f"sports-split"

# print("Loading item feat")
# item_feat_path = f"/data2/wangzhongren/taolin_project/{dataset}_hdf5/norm_item_feat.h5"
# item_feat_dict = h5py.File(item_feat_path,'r')
# item_feat = []
# with open(f"/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{split_dataset}.item", 'r') as read_f:
#     lines = read_f.read().splitlines()
#     print(len(lines))
#     for line in tqdm(lines[1:]):
#         item_id = line.split("\t")[0]
#         if item_id not in item_feat_dict.keys():
#             continue
#         item_feat.append(item_feat_dict[item_id][:])
#         print("!!!!",item_feat_dict[item_id][:].shape)
#         exit()
# item_feat = np.stack(item_feat)
# item_feat = torch.from_numpy(item_feat)
# torch.save(item_feat, f"./feat/{split_dataset}/item_feat_input.pt")
# exit()


item_feat = torch.load(f"./feat/{split_dataset}/item_feat_input.pt")
# rs_item_feat = torch.load(f"./feat/{split_dataset}/rs_item_feat_input.pt")
# rs_item_feat = torch.load(f"./ckpt/{split_dataset}/rs_5000_1_index.pt")
# rs_item_feat = rs_item_feat.view(-1)
rs_item_feat = None

print(f"training with {expert_nums} experts")
if use_wandb:
    wandb.init()
torch.random.manual_seed(seed)
model = SimpleVQVAE(codebook_size=num_codes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, item_feat, rs_item_feat, train_iterations=train_iter)
torch.save(model, f"./ckpt/mrq_{num_codes}_{train_iter}.pt")
model.eval()
indices = infer(model, item_feat, rs_item_feat)
os.makedirs(f"ckpt/{split_dataset}/",exist_ok=True)
print(f"save to ./ckpt/{split_dataset}/moe_{num_codes}_{expert_nums}_index.pt")
torch.save(indices, f"./ckpt/{split_dataset}/moe_{num_codes}_{expert_nums}_index.pt")

