import random

import torch
from torch import nn

def filter(patches, label, config):
    pdist = nn.PairwiseDistance(p=2)
    white_mean = (torch.tensor([1.0, 1.0, 1.0]) - torch.tensor(config.NORMALIZE[0])) / torch.tensor(config.NORMALIZE[1])
    white_std = torch.tensor([0., 0., 0.])
    
    filtered_patches = []
    
    H = patches.size()[0]
    W = patches.size()[1]
    for i in range(H):
        for j in range(W):
            sub_img = patches[i, j]
            if config.MEAN_STD is not None:
                mean = torch.mean(sub_img, dim=[1,2])
                std = torch.std(sub_img, dim=[1,2])
                if (pdist(white_mean, mean) < config.MEAN_STD[0]) or (pdist(white_std, std) < config.MEAN_STD[1]):
                    continue
                else:
                    filtered_patches.append([sub_img, [i, j], label])
            else:
                filtered_patches.append([sub_img, [i, j], label])
    
    return filtered_patches

def get_filtered_patches(img, label, config):
    '''
    Images to patches. Images will be filtered if need.
    Return list that have elements consist of patch, coord, label.
    '''
    patches = img.unfold(1, config.PATCH_SIZE, config.PATCH_SIZE).unfold(2, config.PATCH_SIZE, config.PATCH_SIZE)
    patches = patches.permute(1, 2, 0, 3, 4)

    filtered_patches = filter(patches, label, config)
                
    return filtered_patches


def get_patch_batch_list(filtered_patches_list, config):
    '''
    Gather patches to make patch-batch that will be inputed into extractor.
    '''
    patch_stack = []
    for idx, filtered_patches in enumerate(filtered_patches_list):
        patch_stack.extend([[fp[0], [idx]+fp[1], fp[2]] for fp in filtered_patches])

    if config.SHUFFLE:
        random.shuffle(patch_stack)

    patch_batch_list = []
    last_idx = 0
    for _ in range(len(patch_stack)//config.PATCH_BATCH_SIZE + 1):
        temp = patch_stack[last_idx : last_idx + config.PATCH_BATCH_SIZE]
        if temp:
            patch_batch = [t[0] for t in temp]
            patch_batch = torch.stack(patch_batch)
            coord = [t[1] for t in temp]
            label = [t[2] for t in temp]
            patch_batch_list.append([patch_batch, coord, label])
            last_idx += config.PATCH_BATCH_SIZE

    return patch_batch_list


class MyCollator(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, samples):
        filtered_patches_list = [s[0] for s in samples]
        labels = torch.tensor([s[1] for s in samples])
        patch_batch_list = get_patch_batch_list(filtered_patches_list, self.config)
    
        return patch_batch_list, labels