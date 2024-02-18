import datetime
import os
import argparse
import time

import torch
from torch.optim import lr_scheduler

from dataset import get_classification_datasets
from framework import build_model
from cla_utils.map import mAP
from cla_utils.asl_loss import AsymmetricLossOptimized

def parser_args():

    parser = argparse.ArgumentParser(description="Training in torch 2.0.1")
    
    parser.add_argument('--num_class', default=36, type=int, help="Number of query slots")
    parser.add_argument('--optim', default='AdamW', type=str,)
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight_decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)', dest='weight_decay')

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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    parser.add_argument('--device', default='cuda:0', type=str)
             
    args = parser.parse_args()
    return args

def main():
    
    best_mAP = 0
    args = parser_args()

    # build model
    model = build_model(args).to(args.device)
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
    train_loader, valid_loader, test_loader = get_classification_datasets(args)
    
    print("Data Load Finishing!")

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.2
    )

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        print("Epoch: {}".format(epoch))

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, scheduler, args.device)
        print("train loss: {}".format(train_loss))

        # validation for one epoch
        valid_loss, mAP = validate(valid_loader, model, criterion, args.device)
        print("validate loss: {}, mAP: {}".format(valid_loss, mAP))

        # 保存表现最好的模型
        model_path = "./checkpoints/class"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = args.model_name + '_classification_torch2.pth'
        if best_mAP < mAP:
            best_mAP = mAP
            try:
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            except:
                pass
            

def train(train_loader, model, criterion, optimizer, scheduler, device):
    # switch to train mode
    model.train()

    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()

    for _, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

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
def validate(valid_loader, model, criterion, device):
    # switch to eval mode
    model.eval()

    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()

    targets = []
    outputs = []

    with torch.no_grad():
        for _, (images, target) in enumerate(valid_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            output = torch.sigmoid(output).cpu()
            outputs.append(output)
            targets.append(target.cpu())

            # record loss
            losses.update(loss.item(), images.size(0))

    map = mAP(torch.cat(targets).numpy(), torch.cat(outputs).numpy())
    print("valid time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return losses.avg, map

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
