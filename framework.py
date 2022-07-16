import torch
import torch.nn as nn

import math

from models.backbone import build_backbone
from models.transformer import build_transformer

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
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
    def __init__(self, args, backbone, transfomer, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes.
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.query_embed = nn.Embedding(num_class, hidden_dim)
        # self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # 模型并行
        self.backbone = self.backbone.to('cuda:0')
        self.transformer = self.transformer.to('cuda:1')
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1).to('cuda:1')
        self.query_embed = nn.Embedding(num_class, hidden_dim).to('cuda:1')
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True).to('cuda:1')



    def forward(self, input):
        src, pos = self.backbone(input)
        
        src, pos_embed = src[-1].to('cuda:1'), pos[-1].to('cuda:1')
        query_embed = self.query_embed.weight

        if self.args.backbone in ['ViT']:
            N = src.shape[0]
            Y = int(math.sqrt(src.shape[1] / self.backbone.num_channels))
            src = src.unsqueeze(2).unsqueeze(3)
            src = src.view(N, self.backbone.num_channels, Y, Y)

        hs = self.transformer(self.input_proj(src), query_embed, pos_embed)[0] # B,K,d
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        # print(out.shape)
        return out


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = myModel(
        args = args,
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class)

    return model
        
        
