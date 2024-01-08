# The UPerNet with the backbone swin transformer framenwork
from torch import nn
import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor

from models.FPNhead import FPNHEAD
from framework import build_model

class UPerNet(nn.Module):
    def __init__(self, args):
        super(UPerNet, self).__init__()
        # segmentation nun_classes + background
        self.num_classes = args.num_classes + 1

        model = build_model(args)
        model_path = os.path.join("./checkpoints", args.model_name + "_classification_torch2.pth")
        model.load_state_dict(torch.load(model_path))
        model = nn.Sequential(*list(model.children())[:1])[0][0]
        self.backbone = create_feature_extractor(
            model, {"1.0": "swin1", "1.1": "swin2", "1.2": "swin3", "1.3": "swin4",}
        )
        self.in_channels = 1024
        self.channels = 128
        self.decoder = FPNHEAD(channels=self.in_channels, out_channels=self.channels)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.channels, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x1 = x['swin1'].permute(0, 3, 1, 2) # torch.Size([8, 128, 96, 96])
        x2 = x['swin2'].permute(0, 3, 1, 2) # torch.Size([8, 256, 48, 48])
        x3 = x['swin3'].permute(0, 3, 1, 2) # torch.Size([8, 512, 24, 24])
        x4 = x['swin4'].permute(0, 3, 1, 2) # torch.Size([8, 1024, 12, 12])
        x_list = []
        x_list.append(x1)
        x_list.append(x2)
        x_list.append(x3)
        x_list.append(x4)
        x = self.decoder(x_list)
        # x.shape: torch.Size([B, 128, 96, 96])

        # 线性插值，图像大4倍，通道数不变: torch.Size([B, 128, 96, 96]) -> torch.Size([B, 128, 384, 384])
        x = nn.functional.interpolate(
            x, size=(x.size(2)*4, x.size(3)*4),mode='bilinear', align_corners=True
        )
        x = self.cls_seg(x)
        # x.shape: torch.Size([B, 37, 384, 384])
        return x