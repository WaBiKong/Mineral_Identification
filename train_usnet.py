import argparse

import torch
from torch.optim import lr_scheduler

from USNet import UPerNet
from dataset import get_segmentation_datasets
from seg_utils.seg_train_utils import train_one_epoch as train
from seg_utils.seg_train_utils import evaluate as validate

def parser_args():

    parser = argparse.ArgumentParser(description="train usnet")

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
    parser.add_argument('--batch_size', default=8, type=int)
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

def main():
    
    args = parser_args()
    best_miou = 0.0
    
    train_loader, val_loader, test_loader = get_segmentation_datasets(args=args)
    print("dataset load finash!")

    model = UPerNet(args=args).to(args.device)
    # 冻结backbone部分的权重
    for param in model.backbone.parameters():
        param.requires_grad = False

    # # 查看网络中参与训练的参数
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
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

    device = args.device
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

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
        model_path = "./checkpoints"
        import os
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = 'usnet_mineral_torch2.pth'
        if best_miou < miou:
            best_miou = miou
            try:
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
            except:
                pass

if __name__ == '__main__':
    main()