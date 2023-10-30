from typing import List

from torch import Tensor
from torch import nn
import timm


from .position_encoding import build_position_encoding

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos

def creat_backbone(model_name):
    if (model_name == 'vit'):
        backbone = timm.create_model('vit_base_patch16_384', pretrained=True)
        channels = 1728
    elif (model_name == 'swin'):
        backbone = timm.create_model('swin_base_patch4_window12_384', pretrained=True)
        channels = 576
    
    return backbone, channels


def build_backbone(args):

    position_embedding = build_position_encoding(args)
    
    backbone, channels = creat_backbone(args.model_name)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    
    model = Joiner(backbone, position_embedding)

    model.num_channels = channels
        
    return model