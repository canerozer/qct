# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler, RADistributedSampler
from .objectcxr import ForeignObjectDataset
from .lvot import UKBiobankLVOTDataset


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    elif config.AUG.RA > 1:
        sampler_train = RADistributedSampler(dataset_train, num_replicas=num_tasks,
                                    repetitions=config.AUG.RA, len_factor=config.AUG.RA,
                                    shuffle=True, drop_last=True)
    else:
        # Try also making # of repetitions as 1.
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    elif config.DATA.DATASET == 'OCXR':
        prefix = 'train' if is_train else 'dev'

        meta = config.DATA.DATA_PATH + prefix + ".csv"
        labels = pd.read_csv(meta, na_filter=False)
        print(f'{len(os.listdir(config.DATA.DATA_PATH + prefix))} pics in {config.DATA.DATA_PATH} {prefix}/')
        print(labels['annotation'])

        labels_dict = dict(zip(labels.image_name,
                                     labels.annotation))

        dataset = ForeignObjectDataset(config.DATA.DATA_PATH, datatype=prefix,
                                        labels_dict=labels_dict, transform=transform)
        nb_classes = 2
    elif config.DATA.DATASET == 'LVOT':
        prefix = 'train' if is_train else 'val'

        meta = os.path.join(config.DATA.DATA_PATH, 'labels', prefix+'.txt')

        # train = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/train.txt
        # val = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/val.txt
        # test = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/test.txt

        dataset = UKBiobankLVOTDataset(meta,
                                       datatype=prefix,
                                       transform=transform)
        nb_classes = 2
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_test_dataset(is_test, config, output_loc=False, output_name=False):
    transform = build_transform(False, config)
    if config.DATA.DATASET == 'OCXR':
        prefix = 'test' if is_test else 'dev'

        meta = config.DATA.DATA_PATH + prefix + ".csv"
        labels = pd.read_csv(meta, na_filter=False)
        print(f'{len(os.listdir(config.DATA.DATA_PATH + prefix))} pics in {config.DATA.DATA_PATH} {prefix}/')
        print(labels['annotation'])

        labels_dict = dict(zip(labels.image_name,
                                     labels.annotation))

        dataset = ForeignObjectDataset(config, datatype=prefix,
                                        labels_dict=labels_dict, transform=transform,
                                        output_loc=output_loc, output_name=output_name)
        nb_classes = 2
    elif config.DATA.DATASET == 'LVOT':
        prefix = 'test' if is_test else 'val'

        meta = os.path.join(config.DATA.DATA_PATH, 'labels', prefix+'.txt')

        # train = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/train.txt
        # val = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/val.txt
        # test = /home/ilkay/Documents/caner/datasets/mdai/create_lvot_dataset/jpg_slices/labels/test.txt

        dataset = UKBiobankLVOTDataset(meta,
                                       datatype=prefix,
                                       transform=transform,
                                       output_loc=output_loc,
                                       output_name=output_name,
                                       config=config)
        nb_classes = 2
    else:
        raise NotImplementedError("We only support testing on LVOT and OCXR Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
