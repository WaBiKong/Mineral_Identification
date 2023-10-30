# This is the infer of classification
import datetime
import os
import argparse
import time

import torch

from dataset import get_classification_datasets
from framework import build_model
from cla_utils.map import mAP
from sklearn import metrics

def parser_args():

    parser = argparse.ArgumentParser(description="Infer")
    
    parser.add_argument('--num_class', default=36, type=int, help="Number of query slots")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')

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
    # parser.add_argument('--model_name', default='vit', type=str)
    parser.add_argument('--model_name', default='swin', type=str)
    # if swin, batch_size = 8
    # if vit, batch_size = 16
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    parser.add_argument('--device', default='cuda:1', type=str)
             
    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()

    # build model
    model = build_model(args).to(args.device)
    model_path = os.path.join("./checkpoints", args.model_name + "_classification_torch2.pth")
    model.load_state_dict(torch.load(model_path))
    print("Model Load Finishing!")

    # Data loading
    train_loader, valid_loader, test_loader = get_classification_datasets(args)
    
    print("Data Load Finishing!")

    torch.cuda.empty_cache()

    # validation for one epoch
    mAP, APs = inference(test_loader, model, args.device)
    print("infer mAP: {}".format(mAP))
    APs = APs.tolist() # numpy.ndarry转list
    print(APs)
    with open(os.path.join("./result", 'APs.txt'), 'w') as f:
        f.write("\n".join('%s'%ap for ap in APs)) # 将list中的float转为str后将list整个转为str

@torch.no_grad()
def inference(valid_loader, model, device):
    # switch to eval mode
    model.eval()

    end = time.time()

    targets = []
    outputs = []

    with torch.no_grad():
        for _, (images, target) in enumerate(valid_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)

            output = torch.sigmoid(output).cpu()
            outputs.append(output)
            targets.append(target.cpu())


    map = mAP(torch.cat(targets).numpy(), torch.cat(outputs).numpy())
    APs = metrics.average_precision_score(torch.cat(targets).numpy(), torch.cat(outputs).numpy(), average=None)
    print("infer time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return map, APs

if __name__ == '__main__':
    main()

# torch 1.10.0
# swin transformer
# infer time: 0:04:23
# infer mAP: 87.8105124098069
# vision transformer
# infer time: 0:03:26
# infer mAP: 81.08409230540593

# torch 2.0.1
# swin transformer
# infer time: 0:03:41
# infer mAP: 86.8124796659374
# infer mAP: 86.91555456878392
