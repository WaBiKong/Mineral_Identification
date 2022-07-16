import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from randaugment import RandAugment
from utils.cutout import SLCutoutPIL

import json
from PIL import Image

class MineralDataset(Dataset):
    def __init__(self, anno_path, 
                num_class, 
                input_transform):

        self.path = anno_path
        self.num_class = num_class
        self.transforms = input_transform

        with open(self.path, encoding="gbk") as f:
            samples = json.load(f)
        
        self.images = []
        self.labels = []

        for sample in samples:
            image = sample['fname']
            label = self.load_labels(sample['labels'])
            self.images.append(image)
            self.labels.append(label)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.load_image(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label

    def load_image(self, img_fpath):
        img: "Image.Image" = Image.open(img_fpath)
        return img.convert("RGB")

    def load_labels(self, multi_labels):
        label = np.zeros(self.num_class)
        for l in multi_labels:
            label[int(l)] = 1.0
        return label

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std = [1, 1, 1])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_transforms_list = [transforms.Resize((args.img_size, args.img_size)),
                        RandAugment(),  # https://arxiv.org/pdf/1909.13719.pdf
                        transforms.ToTensor(),
                        normalize] 
    
    try:
        if args.cutout:
            print("Using Cutout!!!")
            train_transforms_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length, cut_fact=args.cut_fact))
    except Exception as e:
        Warning(e)
    
    train_transforms = transforms.Compose(train_transforms_list)
    test_transforms = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])

    train_dataset = MineralDataset('./annotations/train.json', num_class=args.num_class, input_transform=train_transforms)
    valid_dataset = MineralDataset('./annotations/valid.json', num_class=args.num_class, input_transform=test_transforms)
    test_dataset = MineralDataset('./annotations/test.json', num_class=args.num_class, input_transform=test_transforms)

    print("len(train_dataset)", len(train_dataset))
    print("len(valid_dataset)", len(valid_dataset))
    print("len(test_dataset)", len(test_dataset))

    return train_dataset, valid_dataset, test_dataset