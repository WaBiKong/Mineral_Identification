import datetime
import os
import argparse
import time
from sklearn import  metrics

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import get_datasets
from framework import build_model
from utils.map import mAP
from utils.asl_loss import AsymmetricLossOptimized

def parser_args():

    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument('--num_class', default=36, type=int, help="Number of query slots")
    parser.add_argument('--optim', default='AdamW', type=str,)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight_decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('--pretrained', default=False,
                        help="use pre-trained model. default is False.")
    parser.add_argument('--result_path', default='./train_result', type=str)

    # als loss funcation
    parser.add_argument('--eps', default=1e-5, type=float,
                        help="eps for focal loss (default: 1e-5)")
    parser.add_argument('--dtgfl', default=True, 
                        help="disable_torch_grad_focal_loss in ASL Loss funcation")              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help="gamma pos for simplified asl loss")
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help="gamma neg for simplified asl loss")
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')  

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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")

    # backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="img_size=384, lr=1e-5, batch_size=32")
    # parser.add_argument('--backbone', default='ViT', type=str,
    #                     help="img_size=384, lr=1e-5, batch_size=8")
             
    args = parser.parse_args()
    return args

def main():
    best_mAP = 0

    args = parser_args()
    print(args)

    # build model
    model = build_model(args)
    # criterion
    criterion = AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=args.dtgfl,
        clip=args.loss_clip,
        eps=args.eps,
    )
    # optimizer Adamw
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
        ]
    optimizer = getattr(torch.optim, args.optim)(
        param_dicts,
        args.lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    
    print("Model Load Finishing!")

    # Data loading
    train_dataset, valid_dataset, test_dataset = get_datasets(args)

    # only using train_dataset and valid_dataset
    train_loader = DataLoader(
                        train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
                        valid_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    
    print("Data Load Finishing!")

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)

    with open(os.path.join(args.result_path, "yourTrainingResultPath.txt"), 'w') as f:
        f.write("epoch\t\ttrain_loss\t\tvalid_loss\t\tvalid_mAP\n")

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        print("Epoch: {}".format(epoch))

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, scheduler)
        print("train loss: {}".format(train_loss))

        # validation for one epoch
        valid_loss, mAP1, mAP2 = validate(valid_loader, model, criterion)
        print("validate loss: {}, mAP1: {}, mAP2: {}".format(valid_loss, mAP1, mAP2))

        with open(os.path.join(args.result_path, "yourTrainingResultPath.txt"), 'a') as f:
            f.write("{:5d}\t\t{:.7f}\t\t{:.7f}\t\t{:.7f}\n".format(
                epoch, train_loss, valid_loss, mAP2))

        # 保存表现最好的模型
        model_path = "./checkpoints"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = 'yourModelName.pth'
        if best_mAP < mAP1:
            best_mAP = mAP1
            try:
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            except:
                pass
            

def train(train_loader, model, criterion, optimizer, scheduler):
    # switch to train mode
    model.train()

    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()

    for _, (images, target) in enumerate(train_loader):
        images = images.to('cuda:0')
        target = target.to('cuda:1')

        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), images.size(0))

        # one cycle learning rate
        scheduler.step()

    print("training time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return losses.avg

@torch.no_grad()
def validate(valid_loader, model, criterion):
    # switch to eval mode
    model.eval()

    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()

    targets = []
    outputs = []

    with torch.no_grad():
        for _, (images, target) in enumerate(valid_loader):
            images = images.to('cuda:0')
            target = target.to('cuda:1')

            output = model(images)
            loss = criterion(output, target)

            output = torch.sigmoid(output).cpu()
            outputs.append(output)
            targets.append(target.cpu())

            # record loss
            losses.update(loss.item(), images.size(0))

    mAP1 = mAP(torch.cat(targets).numpy(), torch.cat(outputs).numpy())
    mAP2 = metrics.average_precision_score(torch.cat(targets).numpy(), torch.cat(outputs).numpy(), average=None).mean()
    print("valid time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return losses.avg, mAP1, mAP2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



if __name__ == '__main__':
    main()