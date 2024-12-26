import h5py
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

datanames = ['toys-split','beauty-split','sports-split'] 
dataname = datanames[0]
expert_num = 7
method = ['moe','rq','me'][0]
data_root = f"/data2/wangzhongren/taolin_project/dataset/{dataname}"
index_path = f"../moe/ckpt/{dataname}/{method}_256_{expert_num}_index.pt"
# cbsize = int(index_filename.split("_")[1][6:])
dataset_types = ['train','valid','test']

# 对item数据集的特征列进行处理，包括填充缺失值，对类别的分割


sample_filename = f"{dataname}.item"
feat_keys = ['item_id', 'sales_type', 'brand','categories']
sample_path = os.path.join(data_root,sample_filename)
df_feat = pd.read_csv(sample_path, sep='\t', header=0)
df_feat.columns = [col.split(":")[0] for col in df_feat.keys()]
df_feat= df_feat[feat_keys]

index = torch.load(index_path)

# 把index特征融入到item表中
for i in range(expert_num):
    if method == 'me':
        df_feat[f"{method}_cid_{i+1}"] = index.cpu()[:,0]
    else:
        df_feat[f"{method}_cid_{i+1}"] = index.cpu()[:,i]

df_all = pd.DataFrame()
for dataset_type in dataset_types:
    data_filename = f"{dataname}.{dataset_type}.inter"
    data_path = f"{data_root}/{data_filename}"
    df_data = pd.read_csv(data_path, sep='\t', header=0)
    df_data.columns = [col.split(":")[0] for col in df_data.keys()]
    df_data['label'] = (df_data['rating']>3).astype(int)
    df_data = df_data.drop(columns = ['rating'])
    df_data['dataset_type'] = dataset_type
    df_all = pd.concat([df_all, df_data], ignore_index=True)
# 去掉time列
df_all = df_all.drop(columns=['timestamp'])
# 把item特征混合到训练集中
df_all_merged = pd.merge(df_all,df_feat,on='item_id',how='left')

# 使用labelencoder对稀疏特征进行编码
encodered_columns = [col for col in df_all_merged.columns if 'cid' not in col and col != 'label' and col != 'dataset_type']
lbe = LabelEncoder()
for column in encodered_columns:
    df_all_merged[column] = lbe.fit_transform(df_all_merged[column])
columns = [col for col in df_all_merged.columns if col != 'label'] + ['label']
df_all_merged = df_all_merged[columns]
# 保存为csv和h5
for dataset_type in dataset_types: 
    df_split = df_all_merged[df_all_merged['dataset_type'] == dataset_type].drop(columns=['dataset_type'])
    output_path = f"/data2/wangzhongren/taolin_project/FuxiCTR/data/{dataname}-{method}-{expert_num}"
    os.makedirs(output_path,exist_ok = True)
    df_split.to_csv(os.path.join(output_path,f'{dataset_type}.csv'),index=False)
    print(f"{dataset_type}.csv saved")
