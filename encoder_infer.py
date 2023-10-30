import timm
import os
import argparse
import time
import torch
from dataset import get_classification_datasets

def parser_args():

    parser = argparse.ArgumentParser(description="encoder choose")
    
    parser.add_argument('--num_class', default=36, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--multi', type=bool, default=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--device', type=str, default='cuda:1')
             
    args = parser.parse_args()
    return args

def main():

    args = parser_args()
    print(args)

    # Data loading
    train_dataset, valid_dataset, test_dataset = get_classification_datasets(args)
    print("Data Load Finishing!")

    # build model
    # model_name = 'resnet50'
    # model_name = 'efficientnet_b0'
    # model_name = 'resnext50_32x4d'
    # model_name = 'vit_base_patch16_384'
    model_name = 'swin_base_patch4_window12_384'

    model = timm.create_model(model_name=model_name, num_classes=36, pretrained=False)
    model_path = os.path.join("./checkpoints/encoder", model_name + ".pth")
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)
    print("Model load Finishing!")

    ave_time, valid_top1 = validate(test_dataset, model, device=args.device)
    print("top-1: {}, infer time for a img: {} ms".format(valid_top1, ave_time*1000))



@torch.no_grad()
def validate(valid_loader, model, device):
    # switch to eval mode
    model.eval()
    end = time.time()

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

    ave_top1 = accu_num.item() / sample_num

    all_time = int(time.time() - end)
    ave_time = all_time / sample_num
    print("all sample: {}, all time: {}".format(sample_num, all_time))

    return ave_time, ave_top1

if __name__ == "__main__":
    main()

# resnet50
# top-1: 0.7814915623298857, infer time for a img: 13.0647795318454 ms
# top-1: 0.7814915623298857, infer time for a img: 12.955906369080022 ms

# efficientnet_b0
# top-1: 0.7704953728905825, infer time for a img: 6.967882416984214 ms

# resnext50_32x4d
# top-1: 0.7690800217746325, infer time for a img: 18.617310832879692 ms

# vit_base_patch16_384
# top-1: 0.7871529667936854, infer time for a img: 18.072945019052803 ms

# swin_base_patch4_window12_384
# top-1: 0.8265650517147524, infer time for a img: 21.883505715841043 ms