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
    parser.add_argument('--hela', type=bool, default=False)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
             
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
    model_name = 'efficientnet_b0'
    # model_name = 'resnext50_32x4d'
    # model_name = 'vit_base_patch16_384'
    # model_name = 'swin_base_patch4_window12_384'

    model = timm.create_model(model_name=model_name, num_classes=36, pretrained=False)
    model_path = os.path.join("./checkpoints/encoder", model_name)
    if args.hela :
        model_path = model_path + "_HELA.pth"
    else:
        model_path = model_path + ".pth"
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

    # with torch.no_grad():
    #     for step, (images, target) in enumerate(valid_loader):
    #         images = images.to(device)
    #         target = target.to(device)

    #         sample_num += target.shape[0]

    #         output = model(images)
            
    #         pred_classes = torch.max(output, dim=1)[1]
    #         target = torch.max(target,dim=1)[1]

    #         accu_num += torch.eq(pred_classes, target).sum()

    # ave_top1 = accu_num.item() / sample_num

    # all_time = int(time.time() - end)
    # ave_time = all_time / sample_num
    # print("all sample: {}, all time: {}".format(sample_num, all_time))

    # return ave_time, ave_top1

    from sklearn.metrics import confusion_matrix, accuracy_score

    # 初始化混淆矩阵
    num_classes = 36  # 类别数量
    conf_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for step, (images, target) in enumerate(valid_loader):
            images = images.to(device)
            target = target.to(device)
            sample_num += target.shape[0]

            output = model(images)
            pred_classes = torch.max(output, dim=1)[1]
            target = torch.max(target,dim=1)[1]

            accu_num += torch.eq(pred_classes, target).sum()

            # 更新混淆矩阵
            conf_matrix += confusion_matrix(target.cpu(), pred_classes.cpu(), labels=range(num_classes))

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        class_accuracies.append((conf_matrix[i, i] / conf_matrix[i, :].sum()).item())

    # 打印每个类别的准确率
    print("Accuracy for class: ", class_accuracies)
    print("Macro Average Accuracy: ", sum(class_accuracies) / len(class_accuracies))

    # 计算总体准确率
    overall_accuracy = accu_num.item() / sample_num
    print(f'Overall accuracy: {overall_accuracy}')

    all_time = int(time.time() - end)
    ave_time = all_time / sample_num

    return ave_time, overall_accuracy

if __name__ == "__main__":
    main()

# resnet50
# top-1: 0.7814915623298857, infer time for a img: 13.0647795318454 ms
# resnet50 + HELA
# top-1: 0.7642767706461974, infer time for a img: 7.512657194185857 ms

# efficientnet_b0
# top-1: 0.7683597365126028, infer time for a img: 6.967882416984214 ms

# resnext50_32x4d
# top-1: 0.7690800217746325, infer time for a img: 18.617310832879692 ms

# vit_base_patch16_384
# top-1: 0.7871529667936854, infer time for a img: 18.072945019052803 ms

# swin_base_patch4_window12_384
# top-1: 0.8265650517147524, infer time for a img: 21.883505715841043 ms

# resnet50
# Accuracy for class:  [0.7915309446254072, 0.4915254237288136, 0.84375, 0.5766871165644172, 0.9267015706806283, 0.7988950276243094, 0.6153846153846154, 0.46407185628742514, 0.82, 0.7964285714285714, 0.9512195121951219, 0.32919254658385094, 0.7961335676625659, 0.7762039660056658, 0.8645953202915229, 0.7251655629139073, 0.8275862068965517, 0.5952380952380952, 0.6192, 0.44635193133047213, 0.8770864946889226, 0.3227848101265823, 0.7147766323024055, 0.6588235294117647, 0.761501210653753, 0.8296906045983536, 0.8426966292134831, 0.8571428571428571, 0.6857142857142857, 0.6411483253588517, 0.5647425897035881, 0.6869918699186992, 0.6968085106382979, 0.6391184573002755, 0.8543689320388349, 0.8576826196473551]
# Macro Average Accuracy:  0.7096372276083404
# Overall accuracy: 0.7710272742119876

# resnet50 + HELA
# Accuracy for class:  [0.8794788273615635, 0.4011299435028249, 0.8020833333333334, 0.48466257668711654, 0.9280104712041884, 0.7856353591160221, 0.5783475783475783, 0.49101796407185627, 0.7133333333333334, 0.8107142857142857, 0.8902439024390244, 0.453416149068323, 0.8347978910369068, 0.6968838526912181, 0.8638281549673955, 0.6158940397350994, 0.8068965517241379, 0.4166666666666667, 0.712, 0.463519313304721, 0.8679817905918058, 0.37341772151898733, 0.6907216494845361, 0.611764705882353, 0.7227602905569007, 0.8081180811808119, 0.8202247191011236, 0.7272727272727273, 0.7047619047619048, 0.5933014354066986, 0.656786271450858, 0.7154471544715447, 0.7446808510638298, 0.6859504132231405, 0.8252427184466019, 0.8425692695214105]
# Macro Average Accuracy:  0.6949878305066897
# Overall accuracy: 0.7642767706461974