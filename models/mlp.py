import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='relu', use_sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_list = hidden_list

        if act == 'relu':
            act_layer=nn.ReLU
        elif act == 'gelu':
            act_layer=nn.GELU
        elif act == 'swish':
            act_layer = Swish
        else:
            raise NotImplementedError

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(act_layer())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def flops(self): # per pixel
        flops = 0
        if len(self.hidden_list) > 0:
            flops += self.in_dim * self.hidden_list[0]
            for i in range(1, len(self.hidden_list)):
                flops += self.hidden_list[i-1] * self.hidden_list[i]
            flops += self.hidden_list[-1] * self.out_dim
        else:
            flops += self.in_dim * self.out_dim
        return flops