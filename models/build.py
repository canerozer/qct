# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch.nn as nn
from torchvision import models as tvmodels

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == "resnet34":
        model = tvmodels.resnet34(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
    elif model_type == "resnet50":
        model = tvmodels.resnet50(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
    elif model_type == "resnet101":
        model = tvmodels.resnet101(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
    elif model_type == "resnet152":
        model = tvmodels.resnet152(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
    elif model_type == "efficientnetb0":
        model = tvmodels.efficientnet_b0(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb1":
        model = tvmodels.efficientnet_b1(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb2":
        model = tvmodels.efficientnet_b2(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb3":
        model = tvmodels.efficientnet_b3(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb4":
        model = tvmodels.efficientnet_b4(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb5":
        model = tvmodels.efficientnet_b5(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb6":
        model = tvmodels.efficientnet_b6(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    elif model_type == "efficientnetb7":
        model = tvmodels.efficientnet_b7(pretrained=config.MODEL.TORCHVISION_PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
