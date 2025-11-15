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


from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention, LogisticRegression
from fuxictr.pytorch.torch_utils import get_initializer

class AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 use_index_emb=False,
                 index_file_path=None,
                 codebook_size=256,
                 **kwargs):
        super(AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False) if use_wide else None
        self.use_index_emb = use_index_emb
        self.embedding_dim = embedding_dim
        self.num_indices = 0
        if self.use_index_emb:
            assert index_file_path is not None, "使用index embedding时，必须提供index_file_path"
            index_data = torch.load(index_file_path)
            if index_data.dim() == 1:
                index_data = index_data.unsqueeze(1)
            self.num_indices = index_data.shape[1]
            self.codebook_size = codebook_size
            self.register_buffer('index_data', index_data)
            self.index_embedding_layers = nn.ModuleList([
                nn.Embedding(codebook_size, embedding_dim)
                for _ in range(self.num_indices)
            ])
            for index_emb_layer in self.index_embedding_layers:
                embedding_initializer = get_initializer("partial(nn.init.normal_, std=1e-4)")
                embedding_initializer(index_emb_layer.weight)

        # --- 修正：计算包含 index_emb 后的总维度 ---
        total_emb_dim = feature_map.sum_emb_out_dim()
        if self.use_index_emb:
            total_emb_dim += self.num_indices * embedding_dim

        self.dnn = MLP_Block(input_dim=total_emb_dim,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=net_dropout, 
                                     use_residual=use_residual, 
                                     use_scale=use_scale,
                                     layer_norm=layer_norm) \
             for i in range(attention_layers)])
        

        num_total_fields = feature_map.num_fields + self.num_indices
        self.fc = nn.Linear(num_total_fields * attention_dim, 1)
        
        # self.mix = 'moe' in kwargs['dataset_id']
        self.mix = False
        if self.mix:
            input_dim = feature_map.sum_emb_out_dim()
            self.prefix = "base"
            self.gating = nn.Sequential(
                *[
                    nn.Linear(input_dim, 8),
                    nn.GELU(),
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
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, feature_type='categorical', dynamic_emb_dim=False)
        combined_feature_emb = feature_emb
        if self.use_index_emb:
            item_positions = X['item_id'].long() - 1
            index_embs = self._get_index_embeddings(item_positions) # [B, num_indices, D]
            combined_feature_emb = torch.cat([feature_emb, index_embs], dim=1) # [B, num_fields + num_indices, D]

        attention_input = combined_feature_emb
        dnn_input = combined_feature_emb.flatten(start_dim=1)
        if self.mix:
            flat_feature_emb = feature_emb.flatten(start_dim=1)
            flat_feature_emb = flat_feature_emb + self.gating(flat_feature_emb)
            bs = feature_emb.shape[0]
            # feature_emb = feature_emb.reshape(bs, -1, 64)
            feature_emb = flat_feature_emb.reshape(bs, -1, 64)
        attention_out = self.self_attention(attention_input)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(dnn_input)
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
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

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output