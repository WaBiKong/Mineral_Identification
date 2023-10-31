# 用于展示一张图片的分割效果
import argparse
import numpy as np
from PIL import Image
import json

import torch

from USNet import UPerNet

def parser_args():

    parser = argparse.ArgumentParser(description="test demo")
    parser.add_argument('--num_classes', default=36, type=int, help="Number of query slots")

    # Transformer
    parser.add_argument('--enc_layers', default=0, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
 
    # position_embedding
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
   
    # parameter
    parser.add_argument('--model_name', default='swin', type=str)
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    parser.add_argument('--device', default='cuda:0', type=str)
             
    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()

    model = UPerNet(args=args).to(args.device)
    # 权重加载
    import os
    path = os.path.join("./checkpoints", "usnet_mineral_torch2.pth")
    model.load_state_dict(torch.load(path))
    print("model upernet load finash!")

    img_path = "./63087_29_25_1.jpg"
    original_img = Image.open(img_path)
    width, height = original_img.size
    from torchvision import transforms
    data_transform = transforms.Compose([transforms.Resize([384, 384]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0).to(args.device)

    model.eval()
    out = model(img)
    out = out.argmax(1)
    print(out[0].shape)
    for i in range(384):
        for j in range(384):
            if out[0][i][j] != 0:
                print(out[0][i][j])
                break

    with open("./palette.json", "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    back_transform = transforms.Compose([transforms.Resize([height, width])])
    out = back_transform(out)
    prediction = out.squeeze(0)
    prediction = prediction.to("cpu").numpy().astype(np.uint8)
    mask = Image.fromarray(prediction)
    mask.putpalette(pallette)
    mask.save("test_result.png")


if __name__ == '__main__':
    main()