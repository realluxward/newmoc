# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine
from IPython import embed
import logging
import torch.nn.functional as F

class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 use_index_emb=False,  # 是否使用index embedding
                 index_file_path=None,  # index文件路径
                 codebook_size=256,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.use_index_emb = use_index_emb
        self.embedding_dim = embedding_dim
        if self.use_index_emb:
            assert index_file_path is not None, "使用index embedding时，必须提供index_file_path"
            
            # 读取index
            index_data = torch.load(index_file_path)  # [n] or [n, num_indices]
            if index_data.dim() == 1:
                index_data = index_data.unsqueeze(1)

            num_indices = index_data.shape[1]
            self.num_indices = num_indices
            self.codebook_size = codebook_size
            self.register_buffer('index_data', index_data)
        
            # 创建embedding层（vocab_size = codebook_size）
            self.index_embedding_layers = nn.ModuleList([
                nn.Embedding(codebook_size, embedding_dim)
                for _ in range(num_indices)
            ])
        mlp_input_dim = feature_map.sum_emb_out_dim()
        logging.info(f"Initial MLP input dim: {mlp_input_dim}")
        # 假设item_position的embedding不进入MLP，与DCNv2逻辑保持一致
        # mlp_input_dim -= self.embedding_dim 
        if self.use_index_emb:
            # 加上index embedding的维度
            mlp_input_dim += self.num_indices * embedding_dim
        self.mlp = MLP_Block(input_dim=mlp_input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        # self.mix = 'moe' in kwargs['dataset_id']
        self.mix = False
        # self.mix = True
        if self.mix:
            mix_dim = kwargs['mix_dim']
            input_dim = feature_map.sum_emb_out_dim()
            self.prefix = "base"
            self.gating = nn.Sequential(
                *[
                    nn.Linear(input_dim, 4),
                    nn.GELU(),
                    nn.Linear(4, input_dim),
                ]
            )
        else:
            self.prefix = "nomix"
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=True)
        # feature_emb = F.normalize(feature_emb, p=2, dim=-1)
        norm = feature_emb.norm(p=2, dim=1, keepdim=True) + 1e-8  # 防止除零
        feature_emb = 0.002 * feature_emb / norm

        y_pred = self.fm(X, feature_emb)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        
        # 【修改】如果使用index embedding，则进行查找和拼接
        if self.use_index_emb:
            # 从输入中获取original_item_id（h5行号）
            item_positions = X['item_id'].long()  # [batch_size]
            index_embs = self._get_index_embeddings(item_positions)
            # 将index embeddings拼接到扁平化的特征后面
            flat_feature_emb = torch.cat([flat_feature_emb, index_embs], dim=1)
        if self.mix:
            flat_feature_emb = feature_emb.flatten(start_dim=1)
            flat_feature_emb = flat_feature_emb + self.gating(flat_feature_emb)
            bs = feature_emb.shape[0]
            # feature_emb = feature_emb.reshape(bs, -1, 64)
            feature_emb = flat_feature_emb.reshape(bs, -1, 64)
        
        y_pred += self.mlp(flat_feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def get_emb(self, inputs, dump_keys):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer.get_emb_dict(X, dynamic_emb_dim=True)
        result_dict = {}
        for k in dump_keys:
            result_dict[k] = feature_emb_dict[k]
        
        # get cross feature
        # moe_cid_1 = feature_emb_dict['moe_cid_1']
        # moe_cid_2 = feature_emb_dict['moe_cid_2']
        # flat_feature_emb = torch.cat([moe_cid_1, moe_cid_2], dim=1).flatten(start_dim=1)
        # embed()
        return result_dict
    
    def _get_index_embeddings(self, item_positions):
        indices = self.index_data[item_positions.long()]  # [batch_size, num_indices]
        index_embs_list = []
        for i in range(self.num_indices):
            idx_i = indices[:, i]  # [batch_size]，取第i个index
            emb_i = self.index_embedding_layers[i](idx_i)  # [batch_size, embedding_dim]
            index_embs_list.append(emb_i)
        index_embs = torch.cat(index_embs_list, dim=1)  # [batch_size, num_indices * embedding_dim]
        return index_embs
    
    import torch

    def get_inputs_dump(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        
        # 存储原始数据
        X_dict = {}
        # 存储One-Hot编码后的数据
        onehot_dict = {}
        
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
                
            # 获取原始数据
            column_index = self.feature_map.get_column_index(feature)
            raw_data = inputs[:, column_index].to(self.device)
            
            # 存储原始数据
            X_dict[feature] = raw_data
            
            # 生成One-Hot编码
            if spec["type"] == "categorical":
                # 获取该特征的类别数量
                vocab_size = spec["vocab_size"]
                
                # 创建One-Hot编码
                # 确保数据类型为整数
                indices = raw_data.long()
                
                # 处理可能的越界索引
                indices = torch.clamp(indices, 0, vocab_size - 1)
                
                # 创建One-Hot编码
                onehot = torch.nn.functional.one_hot(indices, num_classes=vocab_size)
                onehot_dict[feature] = onehot.float()  # 转换为float类型
            else:
                # 对于数值特征，直接使用原始值并增加一个维度
                onehot_dict[feature] = raw_data.unsqueeze(-1).float()  # 增加一维并转换为float
        
        return X_dict, onehot_dict