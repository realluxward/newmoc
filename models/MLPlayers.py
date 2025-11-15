from torch import nn
import sys
sys.path.insert(0, '/data2/wangzhongren/taolin_project/FuxiCTR/')
from fuxictr.pytorch.torch_utils import get_activation
import torch.nn.functional as F


class MyDropout(nn.Dropout):
    def forward(self, input, p=None):
        running_p =  p if p is not None else self.p
        return F.dropout(input, running_p, self.training, self.inplace)
    
class MLP_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_activation=None, 
                 dropout_rates=0.0, 
                 batch_norm=False, 
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                dense_layers.append(MyDropout(p=dropout_rates[idx]))
        if output_dim is not None: 
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs, p=None):
        for layer in self.dnn:
            if isinstance(layer, MyDropout):
                inputs = layer(inputs, p)
            else:
                inputs = layer(inputs)
        return inputs





