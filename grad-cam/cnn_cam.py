import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

import timm


def main():

    checkpoints = "/home/wbk/Mineral_Identification/checkpoints/encoder"

    # model_name = 'resnet50'
    # model_name = 'efficientnet_b0'
    model_name = 'resnext50_32x4d'

    model = timm.create_model(model_name=model_name, num_classes=36, pretrained=False)
    model_path = os.path.join(checkpoints, model_name + ".pth")
    model.load_state_dict(torch.load(model_path))

    # global_pool前一层
    # resnet50 和 resnext50_32x4d
    target_layers = [model.layer4]
    # efficientnet_b0
    # target_layers = [model.bn2]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0, 0, 0], std = [1, 1, 1])])
    # load image
    img_name = "175027_27"
    target_category = 27

    img_path = img_name + ".jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 384)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)

    img_name = os.path.join("result", img_name)
    img_name = os.path.join(img_name, str(target_category))
    if not os.path.exists(img_name):
        os.makedirs(img_name)

    plt.axis('off')   # 去坐标轴
    plt.xticks([])    # 去 x 轴刻度
    plt.yticks([])    # 去 y 轴刻度
    plt.savefig(os.path.join(img_name, model_name + ".jpg"))


if __name__ == '__main__':
    main()