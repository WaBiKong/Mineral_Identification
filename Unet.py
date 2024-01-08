import segmentation_models_pytorch as smp
import torch

model = smp.Unet('resnet50', encoder_weights='imagenet')

x = torch.rand([8, 3, 384, 384])
y = model(x)
print(y[0][0][0][0])

import argparse
from USNet import UPerNet

def parser_args():

    parser = argparse.ArgumentParser(description="test")

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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    
    # 要分割的矿物图片种类
    parser.add_argument('--num_classes', default=36, type=int, help="Number of query slots")

    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--device', default='cuda:1', type=str)
             
    args = parser.parse_args()
    return args


args = parser_args()
model2 = UPerNet(args)
out = model2(x)


