import torch
from tqdm import tqdm
dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"
llm_feat = torch.load(f"./feat/{split_dataset}/item_feat_input.pt").cpu().numpy()
index = torch.load(f"./ckpt/{split_dataset}/moe_{256}_{1}_index.pt").cpu().numpy()
rs_feat = torch.load(f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/FuxiCTR/model_zoo/DCNv2/feat/emb/sample_base_moe_cid_1_64.pt").cpu().detach().numpy()

import numpy as np
from scipy.stats import entropy
from collections import Counter

# 生成 bs*4096 的原始向量 (假设 bs=1000)
bs = 1000
dim = 4096
original_vectors = llm_feat.reshape(-1, 4096)

# 生成聚类结果 bs*1 (假设聚类为10类)
cluster_results = index.reshape(-1, 1)

# 1. 对于 bs*4096 的原始向量，逐维计算每个维度的熵
def calculate_per_dimension_entropy(data, bins=30):
    total_entropy = 0
    for i in tqdm(range(data.shape[1])):
        # 对每个维度进行直方图计算
        hist, _ = np.histogram(data[:, i], bins=bins)
        hist_prob = hist / np.sum(hist)
        total_entropy += entropy(hist_prob, base=2)
    return total_entropy

# 2. 计算聚类结果的香农熵
def calculate_discrete_entropy(data):
    count = Counter(data.ravel())
    probs = np.array(list(count.values())) / len(data)
    return entropy(probs, base=2)

# 计算并打印两者的熵
original_entropy = calculate_per_dimension_entropy(original_vectors, bins=256)
rs_entropy = calculate_per_dimension_entropy(rs_feat, bins=256)
cluster_entropy = calculate_discrete_entropy(cluster_results)

print(f"LLM feat Entropy (bs*4096): {np.round(original_entropy, 3)}")
print(f"RS feat Entropy {np.round(rs_entropy, 3)}")
print(f"Cluster Results Entropy (bs*1): {np.round(cluster_entropy, 3)}")

# 比较信息量
# if original_entropy > cluster_entropy:
#     print("The original vectors contain more information than the clustered results.")
# else:
#     print("The clustered results contain more information.")