import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from randaugment import RandAugment
import json
from PIL import Image
import os
import seg_utils.transforms as T

class MineralClassificationDataset(Dataset):
    def __init__(self, anno_path, 
                num_class, 
                input_transform,
                multi=True):

        self.path = anno_path
        self.num_class = num_class
        self.transforms = input_transform
        self.multi = multi

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
        if self.multi:
            for l in multi_labels:
                label[int(l)] = 1.0
        else: # 单标签读取
            label[int(multi_labels[0])] = 1.0
        return label

class MineralSegmentationDataset(Dataset):
    def __init__(self, root = "./seg_data", transforms = None,
                img_path = "train.json"):
        # 原始图片路径
        image_dir = os.path.join(root, 'cleaned_minerals')
        # 标注数据路径
        mask_dir = os.path.join(root, 'seg_labels')
        # 图片名称
        with open(os.path.join(root, img_path), encoding="gbk") as f:
            img_names = json.load(f)

        self.images = [os.path.join(image_dir, x + ".jpg") for x in img_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in img_names]
        assert (len(self.images) == len(self.masks))

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def get_classification_datasets(args):
    normalize = transforms.Normalize(mean=[0, 0, 0], std = [1, 1, 1])

    train_transforms_list = [
        transforms.Resize((args.img_size, args.img_size)),
        RandAugment(),  # https://arxiv.org/pdf/1909.13719.pdf
        transforms.ToTensor(),
        normalize
    ] 

    train_transforms = transforms.Compose(train_transforms_list)
    test_transforms = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    # 实例化训练数据集
    train_dataset = MineralClassificationDataset('./annotations/train.json',
                                    num_class=args.num_class,
                                    input_transform=train_transforms,
                                    multi = args.multi)

    # 实例化验证数据集
    valid_dataset = MineralClassificationDataset('./annotations/valid.json',
                                    num_class=args.num_class,
                                    input_transform=test_transforms,
                                    multi = args.multi)
    test_dataset = MineralClassificationDataset('./annotations/test.json',
                                    num_class=args.num_class,
                                    input_transform=test_transforms,
                                    multi = args.multi)

    train_loader = DataLoader(
                        train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
                        valid_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    test_loader = DataLoader(
                        test_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    return train_loader, valid_loader, test_loader

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
    
def get_transform(train):
    base_size = 384
    crop_size = 384

    if train:
        return SegmentationPresetTrain(base_size, crop_size)
    else:
        return SegmentationPresetEval(crop_size)

def get_segmentation_datasets(args):

    train_dataset = MineralSegmentationDataset(root="./seg_data",
                                    transforms=get_transform(train=True),
                                    img_path="train.json")

    val_dataset = MineralSegmentationDataset(root="./seg_data",
                                  transforms=get_transform(train=False),
                                  img_path="val.json")
    test_dataset = MineralSegmentationDataset(root="./seg_data",
                                  transforms=get_transform(train=False),
                                  img_path="val.json")

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader