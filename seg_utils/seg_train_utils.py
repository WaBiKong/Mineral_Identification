import torch
from torch import nn
import seg_utils.matrix as utils
import time
import datetime

def criterion(inputs, target):

    losses = nn.functional.cross_entropy(inputs, target, ignore_index=255)

    return losses


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes + 1)
    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()
    with torch.no_grad():
        for _, (images, target) in enumerate(data_loader):
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)

            # record loss
            losses.update(loss.item(), images.size(0))

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    print("evaluate time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return losses.avg, confmat

def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler):
    model.train()

    losses = AverageMeter('Loss', ':5.3f')
    end = time.time()

    for _, (images, target) in enumerate(data_loader):
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), images.size(0))

        lr_scheduler.step()

    print("training time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return losses.avg

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