from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
from IPython import embed
import numpy as np
compare_num = 7
moe_data_list = defaultdict(list)
moe_label_list = []
with open("data/toys-split-moe-7-seq/test.csv",'r') as read_f:
    lines = read_f.read().splitlines()
    for line in lines[1:]:
        cur_line = line.split(",")
        for i in range(1,1+compare_num):
            moe_data_list[f"moe_cid_{i}"].append(int(cur_line[5+i]))
        moe_label_list.append(int(cur_line[2]))
for i in range(1,compare_num+1):
    moe_data_list[f"moe_cid_{i}"] = np.array(moe_data_list[f"moe_cid_{i}"])
moe_label_list = np.array(moe_label_list)

rq_data_list = defaultdict(list)
rq_label_list = []
with open("data/toys-split-rq-7-seq/test.csv",'r') as read_f:
    lines = read_f.read().splitlines()
    for line in lines[1:]:
        cur_line = line.split(",")
        for i in range(1,1+compare_num):
            rq_data_list[f"rq_cid_{i}"].append(int(cur_line[5+i]))
        rq_label_list.append(int(cur_line[2]))
for i in range(1,compare_num+1):
    rq_data_list[f"rq_cid_{i}"] = np.array(rq_data_list[f"rq_cid_{i}"])
rq_label_list = np.array(rq_label_list)

me_data_list = defaultdict(list)
me_label_list = []
with open("data/toys-split-moe-1-seq/test.csv",'r') as read_f:
    lines = read_f.read().splitlines()
    for line in lines[1:]:
        cur_line = line.split(",")
        for i in range(1,1+compare_num):
            me_data_list[f"me_cid_{i}"].append(int(cur_line[5+i]))
        me_label_list.append(int(cur_line[2]))
for i in range(1,compare_num+1):
    me_data_list[f"me_cid_{i}"] = np.array(me_data_list[f"me_cid_{i}"])
me_label_list = np.array(me_label_list)

moe_result = []
rq_result = []
me_result = []
for i in range(1, compare_num+1):
    print(f"------ cid {i} -----")
    moe_data = moe_data_list[f'moe_cid_{i}']
    rq_data = rq_data_list[f'rq_cid_{i}']
    me_data = me_data_list[f'me_cid_{i}']
    moe_result.append(normalized_mutual_info_score(moe_label_list, moe_data))
    rq_result.append(normalized_mutual_info_score(rq_label_list, rq_data))
    me_result.append(normalized_mutual_info_score(me_label_list, me_data))
    print(f"moe mi: {moe_result[-1]}")
    print(f"rq mi: {rq_result[-1]}")
    print(f"me mi: {rq_result[-1]}")

moe_result = np.array(moe_result)
rq_result = np.array(rq_result)
me_result = np.array(me_result)
# moe_result = np.round(moe_result, 4)
# rq_result = rq.result(rq_)
# print(moe_result)
# print(rq_result)
# print(me_result)
print(normalized_mutual_info_score(moe_data_list[f'moe_cid_{1}'], moe_data_list[f'moe_cid_{2}']))
print(normalized_mutual_info_score(rq_data_list[f'rq_cid_{1}'], rq_data_list[f'rq_cid_{2}']))
print(normalized_mutual_info_score(me_data_list[f'me_cid_{1}'], me_data_list[f'me_cid_{2}']))
