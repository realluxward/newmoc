import torch
from IPython import embed
import h5py
import numpy as np
from  sklearn.metrics import roc_auc_score,accuracy_score, normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
label = torch.load("./feat/id/sample_label_64.pt", map_location="cpu").view(-1)
item_id = torch.load("./feat/id/sample_item_id_64.pt", map_location="cpu").view(-1)
moe_cid_1 = torch.load("./feat/id/sample_moe_cid_1_64.pt", map_location="cpu").view(-1)
moe_cid_2 = torch.load("./feat/id/sample_moe_cid_2_64.pt", map_location="cpu").view(-1)

id_list = [item_id, moe_cid_1, moe_cid_2]
name_list = ["item_id", "moe_cid_1", "moe_cid_2"]
for cur_id, name in zip(id_list, name_list):
    print(f"{name} <-> label")
    print(normalized_mutual_info_score(label, cur_id))
    print("")
for i in range(len(id_list)):
    for j in range(len(id_list)):
        if i == j:
            continue
        id1, id2 = id_list[i], id_list[j]
        hash_id = id1 * 2000 + id2
        print(f"{name_list[i]}, {name_list[j]} <-> label")
        print(normalized_mutual_info_score(hash_id, label))

hash_id = id_list[0] * 2000 * 2000 + id_list[1] * 2000 + id_list[2]
print(f" total <-> label")
print(normalized_mutual_info_score(hash_id, label))

# print("item_id <-> moe_cid_1")
# print(normalized_mutual_info_score(moe_cid_1, item_id))
# print("")
# print("item_id <-> moe_cid_2")
# print(normalized_mutual_info_score(moe_cid_2, item_id))
# print("")
# print("moe_cid_1 <-> moe_cid_2")
# print(normalized_mutual_info_score(moe_cid_2, moe_cid_1))
# print("")

# print("item_id <-> label")
# print(mutual_info_score(label, item_id))
# print("")
# print("moe_cid_1 <-> label")
# print(mutual_info_score(label, moe_cid_1))
# print("")
# print("moe_cid_2 <-> label")
# print(mutual_info_score(label, moe_cid_2))
# print("")
# print("item_id <-> moe_cid_1")
# print(mutual_info_score(moe_cid_1, item_id))
# print("")
# print("item_id <-> moe_cid_2")
# print(mutual_info_score(moe_cid_2, item_id))
# print("")
# print("moe_cid_1 <-> moe_cid_2")
# print(mutual_info_score(moe_cid_2, moe_cid_1))