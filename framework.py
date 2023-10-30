# This is the framework of classification
import torch
import torch.nn as nn

import math

from models.backbone import build_backbone
from models.transformer import build_transformer

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class myModel(nn.Module):
    def __init__(self, backbone, transfomer, num_class):

        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.MLP = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):
        feature, pos = self.backbone(input)

        src = feature[-1]
        pos_embed = pos[-1]
        query_embed = self.query_embed.weight

        # # vision transformer [B, 576, 768]
        # # swin transformer [B, 144, 1024]
        # print("src.shape", src.shape) 
        # print("pos_embed.shape", pos_embed.shape) # [B, 2048, 16, 16]
        N = src.shape[0]
        src = src.unsqueeze(2).unsqueeze(3)
        src = src.reshape(N, self.backbone.num_channels, pos_embed.shape[2], pos_embed.shape[3])
        src = self.input_proj(src)
        # print("src.shape", src.shape) # [B, 2048, 16, 16]
        

        output = self.transformer(src, query_embed, pos_embed) # B,K,d
        hs = output[0]  # 去掉后面的位置编码
        # print(hs.shape) # [B, 1, 36, 2048]
        # print(hs[-1].shape) # [B, 36, 2048]
        out = self.MLP(hs[-1])

        return out


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = myModel(
        backbone = backbone,
        transfomer = transformer,
        num_class = 36)

    return model
        
        
