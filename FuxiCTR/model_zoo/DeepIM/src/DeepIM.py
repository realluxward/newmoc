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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InteractionMachine
from fuxictr.pytorch.torch_utils import get_initializer


class DeepIM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepIM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 im_order=2, 
                 im_batch_norm=False,
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 net_batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 use_index_emb=False,  # 是否使用index embedding
                 index_file_path=None,  # index文件路径
                 codebook_size=256,
                 **kwargs):
        super(DeepIM, self).__init__(feature_map, 
                                     model_id=model_id,
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
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
            for index_emb_layer in self.index_embedding_layers:
                embedding_initializer = get_initializer("partial(nn.init.normal_, std=1e-4)")
                embedding_initializer(index_emb_layer.weight)
                
        self.im_layer = InteractionMachine(embedding_dim, im_order, im_batch_norm)
        mlp_input_dim = feature_map.sum_emb_out_dim()
        if self.use_index_emb:
            # 加上index embedding的维度
            mlp_input_dim += self.num_indices * embedding_dim
        self.dnn = MLP_Block(input_dim=mlp_input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=net_batch_norm) \
                   if hidden_units is not None else None
        self.mix = False
        if self.mix:
            input_dim = feature_map.sum_emb_out_dim()
            self.prefix = "base"
            self.gating = nn.Sequential(
                *[
                    nn.Linear(input_dim, 8),
                    nn.GELU(),
                    # nn.Dropout(p=net_dropout),
                    nn.Linear(8, input_dim),
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
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=False)
        combined_feature_emb = feature_emb
        if self.use_index_emb:
            # 2. 获取index embedding，并塑造成 [B, num_indices, D] 的形状
            item_positions = X['item_id'].long() - 1
            index_embs = self._get_index_embeddings(item_positions) # [B, num_indices, D]
            combined_feature_emb = torch.cat([feature_emb, index_embs], dim=1)
        if self.mix:
            flat_feature_emb = feature_emb.flatten(start_dim=1)
            flat_feature_emb = flat_feature_emb + self.gating(flat_feature_emb)
            bs = feature_emb.shape[0]
            # feature_emb = feature_emb.reshape(bs, -1, 64)
            feature_emb = flat_feature_emb.reshape(bs, -1, 64)
        y_pred = self.im_layer(combined_feature_emb)
        if self.dnn is not None:
            y_pred += self.dnn(combined_feature_emb.flatten(start_dim=1))
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
        # index_embs = torch.cat(index_embs_list, dim=1)  # [batch_size, num_indices * embedding_dim] 原版的
        index_embs = torch.stack(index_embs_list, dim=1)  # [B, num_indices, D] 新版的，加到交叉层的做法
        return index_embs