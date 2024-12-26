import torch
from IPython import embed
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from  sklearn.metrics import roc_auc_score,accuracy_score, normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
label = torch.load("./feat/id/sample_label_64.pt", map_location="cpu").view(-1)
item_id_feat = torch.load("./feat/emb/sample_item_id_64.pt", map_location="cpu")
moe_cid_1_feat = torch.load("./feat/emb/sample_moe_cid_1_64.pt", map_location="cpu")
moe_cid_2_feat = torch.load("./feat/emb/sample_moe_cid_2_64.pt", map_location="cpu")
item_id_feat = item_id_feat.detach().numpy()
moe_cid_1_feat = moe_cid_1_feat.detach().numpy()
moe_cid_2_feat = moe_cid_2_feat.detach().numpy()

n_clusters_list = [50]
mi_list = []
feat_list = [item_id_feat, moe_cid_1_feat, moe_cid_2_feat]
name_list = ["item_id_feat","moe_cid_1_feat", "moe_cid_2_feat"]
for n_clusters in n_clusters_list:
    for feat, name in zip(feat_list, name_list):
        # 使用MiniBatchKMeans进行聚类
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=1027, n_init=3).fit(feat)
        discretization = np.array(km.labels_)
        print(f"{name} <-> label")
        print(normalized_mutual_info_score(label, discretization))