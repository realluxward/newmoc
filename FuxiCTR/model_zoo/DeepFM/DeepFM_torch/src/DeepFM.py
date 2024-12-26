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
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
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
        feature_emb = self.embedding_layer(X)
        if self.mix:
            flat_feature_emb = feature_emb.flatten(start_dim=1)
            flat_feature_emb = flat_feature_emb + self.gating(flat_feature_emb)
            bs = feature_emb.shape[0]
            # feature_emb = feature_emb.reshape(bs, -1, 64)
            feature_emb = flat_feature_emb.reshape(bs, -1, 64)
        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
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
    
