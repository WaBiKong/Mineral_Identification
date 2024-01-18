# The decoder for segmentation
import torch.nn.functional as F
import torch
from torch import nn

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6],num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out

class FPNHEAD(nn.Module):
    def __init__(self, channels=1024, out_channels=128):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels//2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels//4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels//8, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # input_fpn
        # torch.Size([B, 128, 96, 96])
        # torch.Size([B, 256, 48, 48])
        # torch.Size([B, 512, 24, 24])
        # torch.Size([B, 1024, 12, 12])

        x1 = self.PPMHead(input_fpn[-1]) # torch.Size([B, 128, 12, 12])

        # 线性插值，图像大一倍，通道数不变，x.shape: torch.Size([B, 128, 24, 24])
        x = nn.functional.interpolate(x1, size=(x1.size(2)*2, x1.size(3)*2),mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2]) # torch.Size([B, 128, 24, 24])
        x2 = self.Conv_fuse1_(x) # x2.shape: torch.Size([B, 128, 24, 24])

        # x.shape: torch.Size([B, 128, 48, 48])
        x = nn.functional.interpolate(x2, size=(x2.size(2)*2, x2.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x) # x3.shape: torch.Size([B, 128, 48, 48])

        # x.shape: torch.Size([B, 128, 96, 96])
        x = nn.functional.interpolate(x3, size=(x3.size(2)*2, x3.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x) # x4.shape: torch.Size([B, 128, 96, 96])

        # x1.shape: torch.Size([B, 128, 12, 12]) -> torch.Size([B, 128, 96, 96])
        x1 = F.interpolate(x1, x4.size()[-2:],mode='bilinear', align_corners=True)
        # x2.shape: torch.Size([B, 128, 24, 24]) -> torch.Size([B, 128, 96, 96])
        x2 = F.interpolate(x2, x4.size()[-2:],mode='bilinear', align_corners=True)
        # x3.shape: torch.Size([B, 128, 24, 24]) -> torch.Size([B, 128, 96, 96])
        x3 = F.interpolate(x3, x4.size()[-2:],mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4], 1) # torch.Size([B, 512, 96, 96])
        x = self.fuse_all(x)

        return x
    
# GAU
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample  # 是否进行上采样的标志

        # 3x3卷积用于处理低层特征
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)  # 低层特征的批归一化层

        # 1x1卷积用于处理高层特征，将通道数降到与低层特征相同
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)  # 高层特征的批归一化层

        if upsample:
            # 转置卷积用于上采样高层特征
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)  # 上采样过程中的批归一化层
        else:
            # 1x1卷积用于降采样高层特征
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)  # 降采样过程中的批归一化层

        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant category information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with category-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level. Tensor.
        :param fm_mask: Mask feature maps with category-specific information. Optional.
        :return: Fused and possibly upsampled features.
        """
        b, c, h, w = fms_high.shape  # 获取高层特征的形状信息

        # 对高层特征进行全局平均池化，并用1x1卷积处理后进行批归一化
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # 用3x3卷积处理低层特征，并进行批归一化
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        # 将低层特征与加权的高层特征相乘
        fms_att = fms_low_mask * fms_high_gp

        if self.upsample:  # 如果进行上采样
            # 转置卷积进行上采样，然后与加权后的特征相加，并通过批归一化和ReLU激活函数
            out = self.relu(self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            # 1x1卷积进行降采样，然后与加权后的特征相加，并通过批归一化和ReLU激活函数
            out = self.relu(self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out
