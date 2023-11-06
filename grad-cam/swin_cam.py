import os
import math
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

import timm


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, result):
        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result


def main():
    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题
    img_size = 384
    assert img_size % 32 == 0

    model = timm.create_model("swin_base_patch4_window12_384", num_classes=36, pretrained=False)
    path = "/home/wbk/Mineral_Identification/checkpoints/encoder/swin_base_patch4_window12_384.pth"
    model.load_state_dict(torch.load(path, map_location="cpu"))

    target_layers = [model.norm]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0, 0, 0], std = [1, 1, 1])])
    # load image
    img_name = "124674_5_25"
    target_category = 5

    img_path = img_name + ".jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, img_size)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)

    img_name = os.path.join("result", img_name)
    img_name = os.path.join(img_name, str(target_category))
    if not os.path.exists(img_name):
        os.makedirs(img_name)

    plt.axis('off')   # 去坐标轴
    plt.xticks([])    # 去 x 轴刻度
    plt.yticks([])    # 去 y 轴刻度
    plt.savefig(os.path.join(img_name, "./swin_base_patch4_window12_384.jpg"))    


if __name__ == '__main__':
    main()