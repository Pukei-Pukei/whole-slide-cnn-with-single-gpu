import os

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

from .util import *


class CustomDataset(datasets.ImageFolder):
    def __init__(self, config, phase, transform=None):
        path_dict = {'train': config.DATA.PATH.TRAIN_DIR, 
                     'val': config.DATA.PATH.VAL_DIR,
                     'test': config.DATA.PATH.TEST_DIR}

        super().__init__(path_dict[phase], transform=transform)
        self.imgs = self.samples
        self.config = config

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        filtered_patches = get_filtered_patches(sample, target, self.config)

        return filtered_patches, target


class CustomDataset_csv(Dataset):
    def __init__(self, config, phase, transform=None):
        root_dict = {'train': config.DATA.PATH.TRAIN_DIR, 
                     'val': config.DATA.PATH.VAL_DIR,
                     'test': config.DATA.PATH.TEST_DIR}
        csv_path_dict = {'train': config.DATA.PATH.TRAIN_CSV, 
                        'val': config.DATA.PATH.VAL_CSV,
                        'test': config.DATA.PATH.TEST_CSV}
        self.config = config
        classes, class_to_idx = self.find_classes()
        pathes, labels = self.make_dataset(root_dict[phase], csv_path_dict[phase], class_to_idx)
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.pathes = pathes
        self.labels = labels
        self.transform = transform

        df = pd.read_csv(csv_path_dict[phase], usecols=[self.config.DATA.IMG_PATH_COL, self.config.DATA.LABEL_COL])
        self.img_names = list(df[self.config.DATA.IMG_PATH_COL])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        sample = self.pil_loader(self.pathes[index])
        target = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)

        filtered_patches = get_filtered_patches(sample, target, self.config)

        return filtered_patches, target

    def make_dataset(self, root, csv_path, class_to_idx):
        df = pd.read_csv(csv_path, usecols=[self.config.DATA.IMG_PATH_COL, self.config.DATA.LABEL_COL])

        pathes = list(df[self.config.DATA.IMG_PATH_COL])
        labels = list(df[self.config.DATA.LABEL_COL])

        pathes = [os.path.join(root, path) for path in pathes]

        if self.config.DATA.IS_MULTI_LABEL:
            temp_list = []
            labels = [label.split() for label in labels]
            for label in labels:
                temp = [0.0]*self.config.NUM_CLASSES
                for cls in label:
                    temp[class_to_idx[cls]] = 1.0
                temp_list.append(temp)
            labels = temp_list
        else:
            labels = [class_to_idx[label] for label in labels]

        return pathes, labels

    def find_classes(self):
        df = pd.read_csv(self.config.DATA.PATH.TRAIN_CSV, usecols=[self.config.DATA.LABEL_COL])

        classes = []
        for label in df[self.config.DATA.LABEL_COL].unique():
            classes.extend(label.split())

        classes = sorted(list(set(classes)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


def create_dataloaders(config, phase):
    pin_memory = True

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize(config.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(*config.NORMALIZE)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(*config.NORMALIZE)
    ])

    if config.DATA.READ_TYPE == 'csv':
        train_dataset = CustomDataset_csv(config, phase, transform=transform)
        valid_dataset = CustomDataset_csv(config, phase, transform=val_transform)
        test_dataset = CustomDataset_csv(config, phase, transform=val_transform)
    else:
        train_dataset = CustomDataset(config, phase, transform=transform)
        valid_dataset = CustomDataset(config, phase, transform=val_transform)
        test_dataset = CustomDataset(config, phase, transform=val_transform)

    my_collate_fn = MyCollator(config)

    # 데이터 로더 정의
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  num_workers=config.NUM_WORKERS,
                                  collate_fn=my_collate_fn,
                                  shuffle=True,
                                  pin_memory=pin_memory,
                                  drop_last=True)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.VAL_BATCH_SIZE,
                                  num_workers=config.NUM_WORKERS,
                                  collate_fn=my_collate_fn,
                                  shuffle=True,
                                  pin_memory=pin_memory,
                                  drop_last=False)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config.VAL_BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS,
                                 collate_fn=my_collate_fn,
                                 shuffle=False,
                                 pin_memory=pin_memory,
                                 drop_last=False)

    dataloaders = {
        'train': train_dataloader,
        'val': valid_dataloader,
        'test': test_dataloader,
    }

    return dataloaders[phase]