import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

import timm

class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    model = timm.create_model("vit_base_patch16_384", num_classes=36, pretrained=False)
    model_path = "/home/wbk/Mineral_Identification/checkpoints/encoder/vit_base_patch16_384.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.blocks[-1].norm1]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0, 0, 0], std = [1, 1, 1])])
    # load image
    img_name = "124674_5_25"
    target_category = 5
    
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

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))

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
    plt.savefig(os.path.join(img_name, "./vit_base_patch16_384.jpg"))


if __name__ == '__main__':
    main()