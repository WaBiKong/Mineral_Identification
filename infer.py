import datetime
import argparse
import os
import time
from sklearn import  metrics

import torch
from torch.utils.data import DataLoader

from dataset import get_datasets
from framework import build_model
from utils.map import mAP

def parser_args():
    parser = argparse.ArgumentParser(description="Test")
    
    parser.add_argument('--num_class', default=36, type=int, help="Number of query slots")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--pretrained', default=True,
                        help="use pre-trained model. default is False.")

    # data aug
    parser.add_argument('--cutout', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 
    parser.add_argument('--orid_norm', default=True,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # position_embedding
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")

    # parameter
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    # backbone
    # parser.add_argument('--backbone', default='ViT', type=str,
    #                     help="img_size=384, lr=1e-5, batch_size=8")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="img_size=384, lr=1e-5, batch_size=32")                     

    args = parser.parse_args()
    return args

def main():

    args = parser_args()

    # build model
    model = build_model(args).to('cuda:1')
    model_path = os.path.join("./checkpoints/yourModelName.pth")
    print("Using model: ", model_path)
    model.load_state_dict(torch.load(model_path))
    print("Model Load Finishing!")

    # Data loading
    train_dataset, valid_dataset, test_dataset = get_datasets(args)

    test_loader = DataLoader(
                        test_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    print("Data Load Finishing!")

    mAP, APs = validate(test_loader, model)
    print("mAP: {}\n APs: {}".format(mAP, APs))

@torch.no_grad()
def validate(valid_loader, model):
    model.eval()

    end = time.time()

    targets = []
    outputs = []

    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):
            images = images.to('cuda:1')
            target = target.to('cuda:1')

            output = model(images)

            output = torch.sigmoid(output).cpu()
            outputs.append(output)
            targets.append(target.cpu())

    map = mAP(torch.cat(targets).numpy(), torch.cat(outputs).numpy())
    APs = metrics.average_precision_score(torch.cat(targets).numpy(), torch.cat(outputs).numpy(), average=None)
    print("test time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return map, APs



if __name__ == '__main__':
    main()