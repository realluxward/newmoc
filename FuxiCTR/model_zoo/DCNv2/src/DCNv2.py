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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2, CrossNetMix
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
        if self.use_index_emb:
            assert index_file_path is not None, "需要提供index_file_path"
            
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
            

        input_dim = feature_map.sum_emb_out_dim()
        # input_dim -= self.embedding_dim # 为了去掉item_position列
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
        
        # self.mix = 'moe' in kwargs['dataset_id']
        self.mix = False
        # self.mix = 'moe' in kwargs["dataset_id"] or 'rq' in kwargs["dataset_id"]
        if self.mix:
            self.prefix = f"base_{kwargs['dataset_id']}"
            self.gating = nn.Sequential(
            *[
                nn.Linear(input_dim, 32),
                nn.GELU(),
                # nn.Dropout(p=net_dropout),
                nn.Linear(32, input_dim),
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
        if self.use_index_emb:
            # 从输入中获取original_item_id（h5行号）
            item_positions = X['item_id'].long()  # [batch_size]
            index_embs = self._get_index_embeddings(item_positions)
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=True)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        if self.use_index_emb:
            flat_feature_emb = torch.cat([flat_feature_emb, index_embs], dim=1)
        if self.mix:
            bs = feature_emb.shape[0]
            feature_emb = feature_emb.reshape(bs, -1, 64)
            flat_feature_emb = flat_feature_emb + self.gating(flat_feature_emb)

        cross_out = self.crossnet(flat_feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(flat_feature_emb)], dim=-1)
        # final_out = self.norm(final_out)
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def _get_index_embeddings(self, item_positions):
        indices = self.index_data[item_positions.long()]  # [batch_size, num_indices]
        index_embs_list = []
        for i in range(self.num_indices):
            idx_i = indices[:, i]  # [batch_size]，取第i个index
            emb_i = self.index_embedding_layers[i](idx_i)  # [batch_size, embedding_dim]
            index_embs_list.append(emb_i)
        index_embs = torch.cat(index_embs_list, dim=1)  # [batch_size, num_indices * embedding_dim]
        return index_embs
    