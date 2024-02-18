import segmentation_models_pytorch as smp
import torch
import argparse
from torch.optim import lr_scheduler

from dataset import get_segmentation_datasets
from seg_utils.seg_train_utils import train_one_epoch as train
from seg_utils.seg_train_utils import evaluate as validate



def parser_args():

    parser = argparse.ArgumentParser(description="train Unet")

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--img_size', default=384, type=int,
                        help="size of input images")
    
    # 要分割的矿物图片种类
    parser.add_argument('--num_classes', default=36, type=int)

    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--device', default='cuda:1', type=str)
             
    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    best_miou = 0.0
    device = args.device
    
    train_loader, val_loader, test_loader = get_segmentation_datasets(args=args)
    print("dataset load finash!")

    model = smp.Unet(
        'resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=args.num_classes + 1
    ).to(device)
    print("model load finash!")

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.2
    )

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        print()
        print("Epoch: {}".format(epoch))
        train_loss = train(model, optimizer, train_loader, device,lr_scheduler=scheduler)
        print("train loss: {}".format(train_loss))

        valid_loss, confmat = validate(model, val_loader, device=device, num_classes=args.num_classes)
        val_info = str(confmat)
        print("validate loss: {}".format(valid_loss))
        print(val_info)

        miou_result = val_info.split('\n')[-1]
        miou = float(miou_result.split(' ')[-1])

        # 保存表现最好的模型
        model_path = "./checkpoints/seg"
        import os
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = 'Unet_seg.pth'
        if best_miou < miou:
            best_miou = miou
            try:
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            except:
                pass

if __name__ == '__main__':
    main()
