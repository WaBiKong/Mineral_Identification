# encoder的选择
# 从resnet50、efficientnet_b0、resnext50、vision transformer、swin transformer中
# 选择对单标签效果最好的模型最为后续的基础模型
import timm

import datetime
import os
import argparse
import time

import torch
from torch import nn
import torch.optim as optim

from dataset import get_classification_datasets

def parser_args():

    parser = argparse.ArgumentParser(description="encoder choose")
    
    parser.add_argument('--num_class', default=36, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--multi', type=bool, default=False)
    parser.add_argument('--hela', type=bool, default=True)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
	# for vision transformer lr=0.0001
    # parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda:0')
             
    args = parser.parse_args()
    return args

def main():

    best_top1 = 0.0

    args = parser_args()
    print(args)

    # Data loading
    train_dataset, valid_dataset, test_dataset = get_classification_datasets(args)
    print("Data Load Finishing!")

    # build model
    model = timm.create_model("resnet50", num_classes=36, pretrained=True)
    # model = timm.create_model("efficientnet_b0", num_classes=36, pretrained=True)
    # model = timm.create_model("resnext50_32x4d", num_classes=36, pretrained=True)
    # model = timm.create_model("vit_base_patch16_384", num_classes=36, pretrained=True)
    # model = timm.create_model("swin_base_patch4_window12_384", num_classes=36, pretrained=True)
    model = model.to(args.device)
    print("Model load Finishing!")

    loss = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        print("Epoch: ", epoch)
        # train for one epoch
        train_loss, train_top1 = train(train_dataset, model, loss, optimizer, device=args.device)
        print(train_loss, train_top1)

        # validation for one epoch
        valid_loss, valid_top1 = validate(valid_dataset, model, loss, device=args.device)
        print(valid_loss, valid_top1)

        # 保存表现最好的模型
        model_path = "./checkpoints/encoder"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = 'resnet50_HELA.pth'
        # model_name = 'efficientnet_b0.pth'
        # model_name = 'resnext50_32x4d.pth'
        # model_name = 'vit_base_patch16_384.pth'
        # model_name = 'swin_base_patch4_window12_384.pth'
        if best_top1 < valid_top1:
            best_top1 = valid_top1
            try:
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            except:
                pass
            

def train(train_loader, model, loss, optimizer, device):
    # switch to train mode
    model.train()
    end = time.time()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    sample_num = 0  # 累计样本数
    optimizer.zero_grad()

    for step, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        sample_num += target.shape[0]

        output = model(images)
        l = loss(output, target)

        pred_classes = torch.max(output, dim=1)[1]
        target = torch.max(target,dim=1)[1]

        accu_num += torch.eq(pred_classes, target).sum()
        accu_loss += l.detach()
        
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

    ave_loss = accu_loss.item() / (step+1)
    ave_top1 = accu_num.item() / sample_num

    print("training time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return ave_loss, ave_top1

@torch.no_grad()
def validate(valid_loader, model, loss, device):
    # switch to eval mode
    model.eval()
    end = time.time()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    sample_num = 0  # 累计样本数

    with torch.no_grad():
        for step, (images, target) in enumerate(valid_loader):
            images = images.to(device)
            target = target.to(device)

            sample_num += target.shape[0]

            output = model(images)
            
            pred_classes = torch.max(output, dim=1)[1]
            target = torch.max(target,dim=1)[1]

            accu_num += torch.eq(pred_classes, target).sum()

            l = loss(output, target)
            accu_loss += l

    ave_loss = accu_loss.item() / (step + 2)
    ave_top1 = accu_num.item() / sample_num

    print("valid time: {}".format(datetime.timedelta(seconds=int(time.time() - end))))

    return ave_loss, ave_top1

if __name__ == "__main__":
		main()
