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
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
from fuxictr.pytorch.torch_utils import get_initializer
from IPython import embed
import h5py

class DCNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_index_emb=False,  # 是否使用index embedding
                 index_file_path=None,  # index文件路径
                 codebook_size=256, 
                 mix = 0,
                 use_contrastive_loss=False, # 是否启用对比学习
                 lambda_sem=1.0, 
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        # 给预训练的id设置新的emb层
        self.use_index_emb = use_index_emb
        self.embedding_dim = embedding_dim
        # 对比学习相关参数
        self.use_contrastive_loss = use_contrastive_loss
        self.lambda_sem = lambda_sem

        if self.use_index_emb:
            assert index_file_path is not None, "需要提供index_file_path"
            topk_file_path = index_file_path.replace(".pt", "_top_bottom10.pt")
            
            # 读取index
            index_data = torch.load(index_file_path)  # [n] or [n, num_indices]

            # 读取topk item id列表
            topk_index = torch.load(topk_file_path)  # [n,num_indices,20]

            if index_data.dim() == 1:
                index_data = index_data.unsqueeze(1)

            num_indices = index_data.shape[1]
            self.num_indices = num_indices
            self.codebook_size = codebook_size
            self.register_buffer('index_data', index_data)
            self.register_buffer('topk_index', topk_index)
        
            # 创建embedding层（vocab_size = codebook_size）
            self.index_embedding_layers = nn.ModuleList([
                nn.Embedding(codebook_size, embedding_dim)
                for _ in range(num_indices)
            ])
            # init index embedding
            for index_emb_layer in self.index_embedding_layers:
                embedding_initializer = get_initializer("partial(nn.init.normal_, std=1e-4)")
                embedding_initializer(index_emb_layer.weight)
            

        input_dim = feature_map.sum_emb_out_dim()
        # input_dim += self.embedding_dim # concat版本
        if self.use_index_emb:
            input_dim += self.num_indices * embedding_dim

        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(input_dim, num_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        # num_fields = input_dim//64
        # reduced_size = num_fields//4
        # self.excitation = nn.Sequential(
        #     *[
        #         nn.Linear(num_fields, reduced_size, bias=False),
        #         nn.GELU(),
        #         # nn.Linear(reduced_size, reduced_size, bias=False),
        #         # nn.GELU(),
        #         nn.Linear(reduced_size, num_fields, bias=False),
        #         # nn.GELU()
        #     ]
        # )
        
        self.mix = mix
        if self.mix:
            self.gating = nn.Sequential(
            *[
                nn.Linear(input_dim, self.mix),
                nn.GELU(),
                # nn.Dropout(p=net_dropout),
                nn.Linear(self.mix, input_dim),
            ]
            )
        else:
            self.prefix = f"nomix_{kwargs['dataset_id']}"
            # self.prefix = "nomix"

        self.fc = nn.Linear(final_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=True)
        flat_feature_emb = feature_emb.flatten(start_dim=1)

        if self.use_index_emb:
            item_positions = X['item_id'].long() - 1
            index_emb_raw = self._get_index_embeddings(item_positions)
            index_emb_raw = index_emb_raw.flatten(start_dim=1)
            # 拼接原始的 index embedding
            final_emb = torch.cat([flat_feature_emb, index_emb_raw], dim=1)
        else:
            final_emb = flat_feature_emb

        final_out = self._forward_path(final_emb)
        y_pred = self.fc(final_out)
        return {"y_pred": self.output_activation(y_pred)}

    ###2: 重写 add_loss 方法以实现对比学习 ###
    def add_loss(self, inputs):
        if not self.use_contrastive_loss or not self.training:
            return super().add_loss(inputs)

        X = self.get_inputs(inputs)
        y_true = self.get_labels(inputs)
        
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=True)
        flat_feature_emb = feature_emb.flatten(start_dim=1)

        item_positions = X['item_id'].long() - 1
        index_emb_raw = self._get_index_embeddings(item_positions) # shape: [B, num_indices, D]
        index_emb_raw = index_emb_raw.flatten(start_dim=1) # shape: [B, num_indices * D]
        index_emb_pos_3d = self._get_topk_index_embeddings(item_positions, top_k=3) # shape: [B, num_indices, D]
        index_emb_pos = index_emb_pos_3d.flatten(start_dim=1) # shape: [B, num_indices * D]

        emb_raw = torch.cat([flat_feature_emb, index_emb_raw], dim=1)
        emb_pos = torch.cat([flat_feature_emb, index_emb_pos], dim=1)

        final_out_raw = self._forward_path(emb_raw)
        score_raw = self.fc(final_out_raw)
        y_pred_raw = self.output_activation(score_raw)
        final_out_pos = self._forward_path(emb_pos)
        score_pos = self.fc(final_out_pos)

        loss_ctr = self.loss_fn(y_pred_raw, y_true, reduction='mean')
        
        # 对比损失 (Semantic loss)
        sign = (y_true * 2 - 1).float() # 标签为1时sign=1, 标签为0时sign=-1
        loss_pos = F.softplus(-sign * (score_pos - score_raw)).mean()

        # 6. 返回加权总损失
        total_loss = loss_ctr + self.lambda_sem * loss_pos
        return total_loss

    # ### 新增辅助函数: 封装网络主干，方便复用 ###
    def _forward_path(self, x):
        """ 接收拼接好的 embedding，通过网络主干（CrossNet, DNN等） """
        if self.mix:
            x = x + self.gating(x)
        
        cross_out = self.crossnet(x)
        
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(x)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(x)], dim=-1)
        return final_out
    
    def _get_index_embeddings(self, item_positions):
        indices = self.index_data[item_positions.long()]  # [batch_size, num_indices]
       
        index_embs_list = []
        
        for i in range(self.num_indices):
            idx_i = indices[:, i]  # [batch_size]，取第i个index
            emb_i = self.index_embedding_layers[i](idx_i)  # [batch_size, embedding_dim]
            index_embs_list.append(emb_i)
        index_embs = torch.cat(index_embs_list, dim=1)  # [batch_size, num_indices * embedding_dim]
        return index_embs
    
    def _get_topk_index_embeddings(self,item_positions,top_k):
        retrieved_indices_for_batch = self.topk_index[item_positions.long()]
        top_k_indices = retrieved_indices_for_batch[:, :, :top_k].long()  # 形状: [batch_size, num_indices, 3]

        avg_embs_list = []
        for i in range(self.num_indices):
            indices_for_layer_i = top_k_indices[:, i, :]
            retrieved_embs = self.index_embedding_layers[i](indices_for_layer_i)
            avg_emb_i = torch.mean(retrieved_embs, dim=1)
            avg_embs_list.append(avg_emb_i)
        avg_embs = torch.stack(avg_embs_list, dim=1)
        return avg_embs