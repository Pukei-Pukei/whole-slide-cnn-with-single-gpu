import os
import shutil
import argparse
from math import ceil
from datetime import datetime, timezone, timedelta

import torch
from torch import nn
from torch import optim

from config import get_config
from dataloader import create_dataloaders
from model import Extractor, Classifier, FocalLoss
from trainer import train_model, eval_model, infer_model

import matplotlib.pyplot as plt
plt.ioff()

def main():
    parser = argparse.ArgumentParser('Whole Slide Training')
    parser.add_argument('config_path', help='<config>.py file path')
    args = parser.parse_args()

    config = get_config(args)

    torch.backends.cudnn.benchmark = True
    
    # Get time for recorder dir name
    tz = timezone(timedelta(hours=config.TIME_ZONE))
    train_serial = datetime.now(tz=tz).strftime("%Y_%m_%d_%H_%M_%S")
    train_serial = train_serial + '_' + config.MODE

    if config.EXPERIMENT_NAME:
        RECORDER_DIR = os.path.join(config.DATA.PATH.RECORDER, config.EXPERIMENT_NAME, train_serial)
    else:
        RECORDER_DIR = os.path.join(config.DATA.PATH.RECORDER, train_serial)
    os.makedirs(RECORDER_DIR, exist_ok=True)
    # Copy config.py file to RECORDER_DIR for easy management
    shutil.copy(args.config_path, RECORDER_DIR)

    config.DATA.PATH.RECORDER = RECORDER_DIR
    config.freeze()

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = False if config.RESUME_PATH is not None else True
    extractor = Extractor(pretrained=pretrained).to(device)
    classifier = Classifier(config.NUM_CLASSES).to(device)

    if config.DATA.PATH.TRAIN_DIR is not None:
        train_loader = create_dataloaders(config, phase='train')
    else:
        train_loader = None

    if config.DATA.PATH.VAL_DIR is not None:
        val_loader = create_dataloaders(config, phase='val')
    else:
        val_loader = None

    if config.DATA.PATH.TEST_DIR is not None:
        test_loader = create_dataloaders(config, phase='test')
    else:
        test_loader = None

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    if config.DATA.IS_MULTI_LABEL:
        loss_fn = nn.BCEWithLogitsLoss() # FocalLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

    if config.MODE == 'train':
        ext_optimizer = optim.AdamW(extractor.parameters(), lr=config.EXT_LR, weight_decay=config.EXT_WEIGHT_DECAY)
        clf_optimizer = optim.AdamW(classifier.parameters(), lr=config.CLF_LR, weight_decay=config.CLF_WEIGHT_DECAY)

        ext_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(ext_optimizer, 
                                                                T_0=ceil(len(train_loader))*config.EPOCHS, 
                                                                eta_min=config.ETA_MIN)
        clf_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(clf_optimizer, 
                                                                T_0=ceil(len(train_loader))*config.EPOCHS, 
                                                                eta_min=config.ETA_MIN)
    else:
        ext_optimizer, clf_optimizer, ext_scheduler, clf_scheduler = [None] * 4


    assert config.MODE in ['train', 'eval', 'infer']
    MODE_fn = {'train' : train_model, 'eval': eval_model, 'infer': infer_model}
    
    MODE_fn[config.MODE](extractor,
                        classifier,
                        loss_fn,
                        ext_optimizer,
                        clf_optimizer,
                        ext_scheduler,
                        clf_scheduler,
                        device,
                        loaders,
                        config)


if __name__ == '__main__':
    main()