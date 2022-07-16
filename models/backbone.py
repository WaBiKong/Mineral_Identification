from typing import Dict, List
from typing import List

import torch
from torch import Tensor
from torch import nn
import torchvision
from torchvision.models import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter
import timm


from models.position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict,):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input: Tensor):
        xs = self.body(input)
        out: Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        if name in ['resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
                norm_layer=FrozenBatchNorm2d)
            return_layers = {'layer4': "0"}
        elif name in ['MobileNetV2']:
            backbone = mobilenet_v2(pretrained=True)
            return_layers = {'features': "0"}
        elif name in ['Big_Transfer']:
            backbone = timm.create_model('resnetv2_101x1_bitm', pretrained=True, num_classes=1000)
            return_layers = {'stages': "0"}
        else:
            raise NotImplementedError("Unknow name: %s" % name)
        NCDICT = {
            'resnet101': 2048,
            'MobileNetV2': 1280,
            'Big_Transfer': 2048,
        }
        num_channels = NCDICT[name]
        super().__init__(backbone, train_backbone, num_channels, return_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False

    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_backbone(args):

    position_embedding = build_position_encoding(args)

    train_backbone = True
    if args.backbone in ['ViT']:
        backbone = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=18432)
        bb_num_channels = 128
    else:
        return_interm_layers = False
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, False, args.pretrained)
        bb_num_channels = backbone.num_channels

    model = Joiner(backbone, position_embedding, args)
    model.num_channels = bb_num_channels
        
    return model