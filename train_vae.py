import torch
import torch.nn as nn
import sys
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from math import ceil
from tqdm.auto import trange
from models.MLPlayers import MLP_Layer
import os

"""
    这个文件使用来训练vq-vae的一套操作，训练结束之后，会把模型参数保存为pt文件,
    并用训好的codebook，来给每一个样本一个index，保存为pt文件
"""

class SimpleVQVAE(nn.Module):
    def __init__(self,
                 input_dim = 4096,
                 down_units = [512, 256],
                 up_units = [256, 512],
                 hidden_dim = 128,
                 expert_num = 3,
                 codebook_size=256,
                 codebook_dim = 32,
                 commitment_weight=0,
                 use_bn = "False",
                 use_model = "moc"):
        super().__init__()
        self.use_model = use_model
        self.expert_num = expert_num
        self.llm_down = MLP_Layer(input_dim=input_dim,
                                  hidden_units=down_units,
                                  output_dim=hidden_dim,
                                  hidden_activations="ReLU",
                                  batch_norm=use_bn)
        
        self.vq = nn.ModuleList(
            [VectorQuantize(dim=hidden_dim, # 这个dim是输入的维度，也是输出的维度
                            codebook_dim=codebook_dim, # codebook中每一个向量的维度
                            codebook_size = codebook_size, # codebook中的类别数目
                            kmeans_init=True,
                            commitment_weight=commitment_weight # 承诺损失的权重
                            ) for i in range(expert_num)] 
        )

        self.rq = ResidualVQ(num_quantizers = expert_num,
                             codebook_size = codebook_size,
                             codebook_dim=codebook_dim,
                             dim=hidden_dim,
                             kmeans_init=True,
                             commitment_weight=commitment_weight,
                             shared_codebook = False,
                             quantize_dropout = False)
        
        self.llm_up = MLP_Layer(input_dim=hidden_dim,
                                  hidden_units=up_units,
                                  output_dim=input_dim,
                                  hidden_activations="ReLU",
                                  batch_norm=use_bn)
    def forward(self, x):
        x = self.llm_down(x)

        quantized_embeddings = []
        quantized_indices = []
        total_commit_loss = 0
        if self.use_model ==  "moc" or self.use_model ==  "me": #如果是moc，需要手动的对不同专家的结果进行聚合，emb采用mean，index堆叠，loss加和
            if self.use_model == "me":
                self.expert_num = 1
            for i in range(self.expert_num):
                quantized_embedding, indices_i, commit_loss_i = self.vq[i](x)
                
                quantized_embeddings.append(quantized_embedding)
                quantized_indices.append(indices_i)
                
                total_commit_loss += commit_loss_i

            quantized_embeddings = torch.stack(quantized_embeddings, dim=1)
            x = quantized_embeddings.mean(dim=1)
            quantized_indices = torch.stack(quantized_indices, dim=1)
        elif self.use_model == "rq":
            quantized_embeddings, quantized_indices, total_commit_loss = self.rq(x) 
            #这里的量化emb是所有层的加和，indices是已经堆叠起来的结果，loss是所有层loss堆叠的结果
            total_commit_loss = total_commit_loss.sum(dim=-1) # 对loss进行加和
            x = quantized_embeddings 
        else:
            raise ValueError("use_model must in 'moc','rq','me'!")
        x = self.llm_up(x)
        return x.clamp(-1, 1), quantized_indices, total_commit_loss #clamp操作可以不用吗，还是说大家都不用还是都用


def train(model, opt, item_feat, epochs, batch_size, alpha, device):
    model.to(device)
    item_feat = item_feat.to(device)
    
    num_samples = item_feat.shape[0]
    num_batches = ceil(num_samples / batch_size)
    
    pbar = trange(num_batches*epochs, desc=f"Training Progress")
    for epoch in range(epochs):
        model.train()
        total_reconstruction_loss = 0.0
        total_cmt_loss = 0.0
        
        perm = torch.randperm(num_samples)
        item_feat = item_feat[perm]
        
        for batch_idx in range(num_batches):
            opt.zero_grad()
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x = item_feat[start_idx:end_idx]
            
            out, indices, cmt_loss = model(x)
            
            reconstruction_loss = (out - x).abs().mean() * 100 #这里的100还需要考量
            cmt_loss = cmt_loss.mean()
            
            loss = reconstruction_loss + alpha * cmt_loss
            loss.backward()
            opt.step()
            
            total_reconstruction_loss += reconstruction_loss.item()
            total_cmt_loss += cmt_loss.item()
            
            pbar.update(1)
            pbar.set_postfix({
                "reconstruction_loss": f"{reconstruction_loss.item():.5f}",
                "cmt_loss": f"{cmt_loss.item():.5f}"
            })
        
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_cmt_loss = total_cmt_loss / num_batches
        
        print(f"Epoch {epoch+1}/{epochs} - Avg Reconstruction Loss: {avg_reconstruction_loss:.5f}, Avg Commitment Loss: {avg_cmt_loss:.5f}")
    
    return model

def infer(model, item_feat):
    x = item_feat.to(device)
    _, indices, _ = model(x)
    return indices


if __name__ == '__main__':

    # python train_vae.py 1 100 1 "moc" "toys-split"
    # 对应的分别是 expert_num epochs device 
    # 可调超参
    lr = 3e-4
    seed = 1027
    shuffle = True
    batch_size = 5000
    expert_num = int(sys.argv[1])
    epochs = int(sys.argv[2])
    device = f"cuda:{sys.argv[3]}"
    torch.random.manual_seed(seed)

    # 模型相关----重要超参
    use_model = sys.argv[4] # "moc" "rq" "me" me对应的就是moc的expert_num=1的情况
    codebook_size = 256 # 类别数目
    codebook_dim = 32 #codebook内部的向量长度
    alpha = 1
    commitment_weight = 0
    down_units = [512, 256]
    up_units = down_units[::-1]
    hidden_dim = 128
    use_bn = "False"

    # 数据集相关
    split_dataset = sys.argv[5]

    item_feat = torch.load(f"/data2/wangzhongren/taolin_project/dataset/{split_dataset}/item_feat_input.pt")
    print(f"training with {expert_num} experts")
    

    model = SimpleVQVAE(down_units = down_units,
                 up_units = up_units,
                 hidden_dim = hidden_dim,
                 expert_num = expert_num,
                 codebook_size = codebook_size,
                 codebook_dim = codebook_dim,
                 use_bn = use_bn,
                 use_model = use_model,
                 commitment_weight=commitment_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, opt, item_feat, epochs, batch_size, alpha, device)
    torch.save(model, f"/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_epoch{epochs}.model") #之后可以通过这里的命名格式，来跑更多的实验，保存更多的模型
    model.eval()
    indices = infer(model, item_feat)
    print(f"save to /data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_epoch{epochs}_index.pt")
    torch.save(indices, f"/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_epoch{epochs}_index.pt")