import sys
import numpy as np
method = sys.argv[1]
root = f"/apdcephfs_cq10/share_1362653/taolinzhang/rs/FuxiCTR/model_zoo/{method}/checkpoints/run"
auc_list = []
if len(sys.argv) >=4:
    prefix = sys.argv[2]
    postfix = sys.argv[3]
else:
    prefix = sys.argv[2]
    postfix = None

# dataset = "toys"
# dataset = "sports"
# dataset = "beauty"

seeds = [1027, 2024, 2333]
# seeds = [1027, 2024]
for i in range(8):
    avg_acc = []
    for seed in seeds:
        if i == 0:
            path = f"{root}/split-{prefix}-{1}/seed_{seed}/split-{prefix}-{1}.log"
        # elif i==1 and postfix is None:
            # path = f"{root}/toys-split-moe-1/seed_{seed}/toys-split-moe-1.log"
        else:
            if postfix == None:
                path = f"{root}/split-{prefix}-{i}/seed_{seed}/split-{prefix}-{i}.log"
            else:
                path = f"{root}/split-{prefix}-{i}-{postfix}/seed_{seed}/split-{prefix}-{i}-{postfix}.log"
        with open(path, 'r') as read_f:
            lines = read_f.read().splitlines()
            try:
                cur_auc = float(lines[-1].split("AUC: ")[1][:7])
            except Exception:
                print(path)
                exit()
            avg_acc.append(cur_auc)
    avg_acc = sum(avg_acc)/len(avg_acc)    
    auc_list.append(round(avg_acc,4))
for auc in auc_list:
    print(auc)
