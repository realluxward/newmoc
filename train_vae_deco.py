import torch
import torch.nn as nn
import sys
from vector_quantize_pytorch import ResidualVQ
from my_vq_impl import MyVectorQuantize, MyEuclideanCodebook, my_gumbel_sample

from math import ceil
from tqdm.auto import trange
from models.MLPlayers import MLP_Layer
import os


class VQExpert(nn.Module):
    def __init__(self, in_feat, hidden, out_feat):
        super().__init__()
        self.vq = VectorQuantize(dim=hidden, codebook_dim=32, codebook_size=256, kmeans_init=True, commitment_weight=0)
        self.down = nn.Linear(in_feat, hidden)
        self.up = nn.Linear(hidden, out_feat)

    def forward(self, x):
        x = self.down(x)
        x, indices, commit_loss = self.vq(x)
        x = self.up(x)
        return x.clamp(-1, 1), indices, commit_loss

def decorrelation_loss_fn(quantized_embeddings):
    """
    计算专家输出嵌入之间的去相关性损失。
    Args:
        quantized_embeddings (Tensor): 形状为 [batch_size, expert_num, hidden_dim] 的张量。
    Returns:
        Tensor: 一个标量，表示去相关性损失。
    """
    batch_size, expert_num, hidden_dim = quantized_embeddings.shape

    if expert_num <= 1:
        return torch.tensor(0.0, device=quantized_embeddings.device)

    # 1. 中心化：减去每个专家在 batch 维度上的均值
    embeddings_mean = quantized_embeddings.mean(dim=0, keepdim=True)
    centered_embeddings = quantized_embeddings - embeddings_mean

    # 2. 调整形状以便计算协方差矩阵
    centered_embeddings = centered_embeddings.permute(1, 0, 2).reshape(expert_num, -1)

    # 3. 计算协方差矩阵
    covariance_matrix = torch.matmul(centered_embeddings, centered_embeddings.T) / (batch_size * hidden_dim - 1)

    # 4. 计算相关性矩阵
    std_dev = torch.sqrt(torch.diag(covariance_matrix))
    std_dev = torch.where(std_dev > 1e-8, std_dev, torch.tensor(1.0, device=std_dev.device))
    correlation_matrix = covariance_matrix / torch.outer(std_dev, std_dev)

    # 5. 损失是相关性矩阵的非对角线元素的平方和
    off_diagonal_loss = correlation_matrix.pow(2).sum() - torch.diagonal(correlation_matrix.pow(2)).sum()
    
    return off_diagonal_loss



class SimpleVQVAE(nn.Module):
    def __init__(self,
                 inference_topk=1,
                 input_dim = 4096,
                 down_units = [512, 256],
                 up_units = [256, 512],
                 hidden_dim = 128,
                 expert_num = 3,
                 codebook_size=256,
                 codebook_dim = 32,
                 commitment_weight=0,
                 use_bn = "False",
                 use_model = "moc",
                 mask_ratio = 0,
                 dropout_rates = 0
                 ):
        super().__init__()
        self.use_model = use_model
        self.expert_num = expert_num

        if isinstance(mask_ratio, (int, float)):
            self.mask_ratios = [0.1*mask_ratio] * expert_num
        elif isinstance(mask_ratio, list):
            if len(mask_ratio) != expert_num:
                raise ValueError(f"The length of mask_ratio list ({len(mask_ratio)}) "
                                 f"must be equal to expert_num ({expert_num}).")
            self.mask_ratios = [0.1 * r for r in mask_ratio]
        else:
            raise TypeError("mask_ratio must be a float, int, list")
        
        if isinstance(dropout_rates, (int, float)):
            self.dropout_rates = [0.1 * dropout_rates] * expert_num
        elif isinstance(dropout_rates, list):
            if len(dropout_rates) != expert_num:
                raise ValueError(f"dropout_rates list length mismatch expert_num")
            self.dropout_rates = [0.1 * r for r in dropout_rates]
        else:
            raise TypeError("dropout_rates must be a float, int, or list")

        self.llm_down = MLP_Layer(input_dim=input_dim,
                                  hidden_units=down_units,
                                  output_dim=hidden_dim,
                                  hidden_activations="ReLU",
                                  batch_norm=use_bn,
                                  dropout_rates=self.dropout_rates
                                  )
        
        if 'moc' in use_model or 'me' in use_model:
            if 'moc-taolin' == use_model:
                self.vq = nn.ModuleList([VQExpert(in_feat=128, hidden=128, out_feat=128) for i in range(expert_num)])
            else:
                self.vq = nn.ModuleList(
                    [MyVectorQuantize(dim=hidden_dim, # 这个dim是输入的维度，也是输出的维度
                                    inference_topk=inference_topk,  # sim_level at inference stages
                                    codebook_dim=codebook_dim, # codebook中每一个向量的维度
                                    codebook_size = codebook_size, # codebook中的类别数目
                                    kmeans_init=True,
                                    commitment_weight=commitment_weight # 承诺损失的权重
                                    ) for i in range(expert_num)] 
            )
        if use_model == 'rq':
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
        quantized_embeddings = []
        quantized_indices, quantized_indices_topk = [], []
        total_commit_loss = 0
        

        decorrelation_loss = torch.tensor(0.0, device=x.device)


        if "moc" in self.use_model or "me" in self.use_model:
            if self.use_model == "me":
                self.expert_num = 1
            for i in range(self.expert_num):
                current_dropout_rate = self.dropout_rates[i]
                if self.training and current_dropout_rate > 0:
                    x_for_expert = self.llm_down(x, p=current_dropout_rate)
                else:
                    latent_x = self.llm_down(x, p=0) 
                    current_mask_ratio = self.mask_ratios[i]
                    if self.training and current_mask_ratio > 0:
                        noisy_x = latent_x.clone() 
                        mask = torch.rand(noisy_x.shape, device=noisy_x.device) < current_mask_ratio
                        noisy_x[mask] = 0.0
                        x_for_expert = noisy_x
                    else:
                        x_for_expert = latent_x
                if self.training:
                    quantized_embedding, indices_i, commit_loss_i = self.vq[i](x_for_expert)
                else:
                    quantized_embedding, indices_i, indices_i_topk = self.vq[i].my_forward_for_inference(x_for_expert)
                    commit_loss_i = 0
                quantized_embeddings.append(quantized_embedding)
                quantized_indices.append(indices_i)
                
                if not self.training:
                    quantized_indices_topk.append(indices_i_topk[0])
                total_commit_loss += commit_loss_i

            quantized_embeddings = torch.stack(quantized_embeddings, dim=1)
            quantized_indices = torch.stack(quantized_indices, dim=1)
            if not self.training:
                quantized_indices_topk = torch.stack(quantized_indices_topk, dim=1)

            # <--- MODIFICATION START: 计算去相关性损失
            if self.training:
                decorrelation_loss = decorrelation_loss_fn(quantized_embeddings)
                reconstructed_latent = quantized_embeddings.mean(dim=1)
        elif self.use_model == "rq":
            latent_x = self.llm_down(x)
            reconstructed_latent, quantized_indices, total_commit_loss = self.rq(latent_x) 
            total_commit_loss = total_commit_loss.sum(dim=-1)
        else:
            raise ValueError("use_model must in 'moc','rq','me'!")
        
        if self.training:
            reconstructed_x = self.llm_up(reconstructed_latent)
        
        # <--- MODIFICATION START: 在返回值中加入 decorrelation_loss
        if self.training:
            return reconstructed_x.clamp(-1, 1), quantized_indices, total_commit_loss, decorrelation_loss
        else:
            return quantized_embeddings, quantized_indices, quantized_indices_topk
        # <--- MODIFICATION END


# <--- MODIFICATION START: 修改 train 函数以接收和使用新损失
def train(model, opt, item_feat, epochs, batch_size, alpha, beta, device):
# <--- MODIFICATION END
    model.to(device)
    item_feat = item_feat.to(device)
    
    num_samples = item_feat.shape[0]
    num_batches = ceil(num_samples / batch_size)
    
    pbar = trange(num_batches*epochs, desc=f"Training Progress")
    for epoch in range(epochs):
        model.train()
        total_reconstruction_loss = 0.0
        total_cmt_loss = 0.0
        # <--- MODIFICATION START: 初始化新损失的记录
        total_decor_loss = 0.0
        # <--- MODIFICATION END
        
        perm = torch.randperm(num_samples)
        item_feat = item_feat[perm]
        
        for batch_idx in range(num_batches):
            opt.zero_grad()
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x = item_feat[start_idx:end_idx]
            
            # <--- MODIFICATION START: 接收新的损失值
            out, indices, cmt_loss, decor_loss = model(x)
            # <--- MODIFICATION END
            
            reconstruction_loss = (out - x).abs().mean() * 100
            cmt_loss = cmt_loss.mean()
            # <--- MODIFICATION START: decor_loss 已经是标量，直接使用
            decor_loss = decor_loss 
            
            # 更新总损失函数
            loss = reconstruction_loss + alpha * cmt_loss + beta * decor_loss
            # <--- MODIFICATION END
            loss.backward()
            opt.step()
            
            total_reconstruction_loss += reconstruction_loss.item()
            total_cmt_loss += cmt_loss.item()
            # <--- MODIFICATION START: 累加新损失
            total_decor_loss += decor_loss.item()
            # <--- MODIFICATION END
            
            pbar.update(1)
            # <--- MODIFICATION START: 更新进度条显示
            pbar.set_postfix({
                "recon_loss": f"{reconstruction_loss.item():.4f}",
                "cmt_loss": f"{cmt_loss.item():.4f}",
                "decor_loss": f"{decor_loss.item():.4f}"
            })
            # <--- MODIFICATION END
        
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_cmt_loss = total_cmt_loss / num_batches
        # <--- MODIFICATION START: 计算并打印平均新损失
        avg_decor_loss = total_decor_loss / num_batches
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg Recon Loss: {avg_reconstruction_loss:.5f}, Avg Cmt Loss: {avg_cmt_loss:.5f}, Avg Decor Loss: {avg_decor_loss:.5f}")
        # <--- MODIFICATION END
    
    return model

def infer(model, item_feat):
    x = item_feat.to(device)
    model.eval()
    _, indices, indices_topk = model(x)
    return indices, indices_topk


if __name__ == '__main__':

    # python train_vae.py 1 100 1 "moc" "toys-split" 1 0 0.1
    # python train_vae.py [expert_num] [epochs] [device] [use_model_quantize] [dataset] [mask_ratio] [dropout_ratio] [decorrelation_weight]
    # 对应的分别是 expert_num epochs device ...
    lr = 3e-4
    seed = 1027
    shuffle = True
    batch_size = 5000
    expert_num = int(sys.argv[1])
    epochs = int(sys.argv[2])
    device = f"cuda:{sys.argv[3]}"
    torch.random.manual_seed(seed)

    # 模型相关----重要超参
    use_model = sys.argv[4]
    codebook_size = 256
    codebook_dim = 32
    alpha = 1
    commitment_weight = 0
    down_units = [512, 256]
    up_units = down_units[::-1]
    hidden_dim = 128
    use_bn = "False"
    
    # 推理相关
    inference_topk = 10

    mask_ratio_str = sys.argv[6]
    if len(mask_ratio_str) > 1:
        if len(mask_ratio_str) != expert_num:
            raise ValueError(f"Mask ratio string length must match expert_num.")
        mask_input = [int(char) for char in mask_ratio_str]
    else:
        mask_input = int(mask_ratio_str)
    
    dropout_ratio_str = sys.argv[7]
    if len(dropout_ratio_str) > 1:
        if len(dropout_ratio_str) != expert_num:
            raise ValueError(f"Dropout ratio string length must match expert_num.")
        dropout_input = [int(char) for char in dropout_ratio_str]
    else:
        dropout_input = int(dropout_ratio_str)

    # <--- MODIFICATION START: 添加新的超参数 beta
    beta = float(sys.argv[8]) # 去相关性损失的权重
    # <--- MODIFICATION END

    # 数据集相关
    split_dataset = sys.argv[5]

    item_feat = torch.load(f"/data2/wangzhongren/taolin_project/dataset/{split_dataset}/item_feat_input.pt")
    print(f"training with {expert_num} experts, decorrelation weight (beta): {beta}")
    
    model = SimpleVQVAE(down_units = down_units,
                inference_topk = inference_topk,
                 up_units = up_units,
                 hidden_dim = hidden_dim,
                 expert_num = expert_num,
                 codebook_size = codebook_size,
                 codebook_dim = codebook_dim,
                 use_bn = use_bn,
                 use_model = use_model,
                 commitment_weight=commitment_weight,
                 mask_ratio=mask_input,
                 dropout_rates=dropout_input).to(device)

    # save_prefix = f'/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_mask{mask_ratio_str}_drop{dropout_ratio_str}_beta{beta}_epoch{epochs}'
    # save_prefix = f'/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_mask{mask_ratio_str}_epoch{epochs}'
    save_prefix = f'/data2/wangzhongren/taolin_project/dataset/{split_dataset}/{use_model}_cbsize{codebook_size}_cbdim{codebook_dim}_scala{expert_num}_epoch{epochs}'

    run_mode = sys.argv[-1]
    if run_mode == 'train':
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        train(model, opt, item_feat, epochs, batch_size, alpha, beta, device)    
        torch.save(model, f"{save_prefix}.model")
    elif run_mode == 'infer':
        raw_model = torch.load(f"{save_prefix}.model", map_location='cpu').to(device)
        
        model.load_state_dict(raw_model.state_dict(), strict=False)

    model.eval()
    indices, indices_topk = infer(model, item_feat)
    torch.save(indices, f"{save_prefix}_index.pt")
    torch.save(indices_topk, f"{save_prefix}_index_top_bottom{inference_topk}.pt")