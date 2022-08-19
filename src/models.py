"""
Copyright (c) Facebook, Inc.
Released under the MIT license
https://github.com/facebookresearch/r3m/blob/main/LICENSE
"""
import torch
from torch.utils import data as data
from torchvision import transforms as transforms
import torch.nn as nn

# https://github.com/facebookresearch/r3m/blob/eval/evaluation/r3meval/utils/gaussian_mlp.py
# https://github.com/facebookresearch/r3m/blob/eval/evaluation/r3meval/utils/fc_network.py
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 ):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )

        # Batch Norm Layers
        self.bn = torch.nn.BatchNorm1d(obs_dim)

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh

    def forward(self, x):
        # Small MLP runs on CPU
        # Required for the way the Gaussian MLP class does weight saving and loading.
        # if x.is_cuda:
        #     out = x.to('cpu')
        # else:
        #     out = x
        out = x
        
        ## BATCHNORM
        out = self.bn(out)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out