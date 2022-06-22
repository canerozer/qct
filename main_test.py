# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from cProfile import label
import os
import time
import random
import argparse
from argparse import Namespace
import datetime
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_test_dataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm,
                   list_all_checkpoints, find_best_checkpoint, 
                   reduce_tensor, calculate_roc_auc)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    from torch.cuda import amp
    # amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--eval_all_models', action='store_true', default=False,
                        help='Evaluate on all of the available models given a config')
    parser.add_argument('--debug', action='store_true',
                        help='Entering the debug mode')
    parser.add_argument('--seed', type=int, help='Define seed', default=0)


    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    sal_arg_names = ['eval_all_models', 'debug']
    args_aux = {k: v for k,v in vars(args).items() if k in sal_arg_names}
    args = {k: v for k,v in vars(args).items() if k not in sal_arg_names}

    args = Namespace(**args)
    args_aux = Namespace(**args_aux)

    config = get_config(args)

    return args, args_aux, config


def main(config, args_aux):
    config.defrost()
    dataset_val, config.MODEL.NUM_CLASSES = build_test_dataset(False, config, output_name=True)
    dataset_test, config.MODEL.NUM_CLASSES = build_test_dataset(True, config, output_name=True)
    config.freeze()

    if config.TEST.SEQUENTIAL:
        sampler = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model.eval()
    logger.info(str(model))


    # PASSS
    # optimizer = build_optimizer(config, model)
    # scaler = None
    # scaler_flag = None
    # try:
    #     if config.AMP_OPT_LEVEL != "O0":
    #         model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # except AttributeError:
    #     scaler_flag = True
    #     scaler = amp.GradScaler()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    else:
        input = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).cuda()
        flops = FlopCountAnalysis(model, input)
        logger.info(f"number of GFLOPs: {flops.total() / 1e9}")

    # lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    criterion = torch.nn.CrossEntropyLoss()

    # max_accuracy = 0.0

    # if config.TRAIN.AUTO_RESUME:
    #     resume_file = auto_resume_helper(config.OUTPUT)
    #     if resume_file:
    #         if config.MODEL.RESUME:
    #             logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
    #         config.defrost()
    #         config.MODEL.RESUME = resume_file
    #         config.freeze()
    #         logger.info(f'auto resuming from {resume_file}')
    #     else:
    #         logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    logger.info(f"==============> Loading Model {config.MODEL.NAME}....................")
    if args_aux.eval_all_models:
        checkpoint_paths = list_all_checkpoints(config.OUTPUT)
    else:
        checkpoint_paths = find_best_checkpoint(config.OUTPUT)
    result_dump = {'epoch':[], 'acc_val':[], 'auc_val':[], 'loss_val':[],
                   'acc_test':[], 'auc_test':[], 'loss_test':[], 'throughput':[]}
    thpt = None
    stats_val = None
    stats_test = None
    for d, checkpoint_path in enumerate(checkpoint_paths):
    
        epoch = os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]
        checkpoint = torch.load(os.path.join(config.OUTPUT, checkpoint_path),
                                map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        
        if d == 0 and not args_aux.debug:
            thpt = throughput(data_loader_val, model, logger)

        acc_val, auc_val, loss_val, stats_val = validate(config, data_loader_val, model)
        acc_test, auc_test, loss_test, stats_test = validate(config, data_loader_test, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {acc_val:.3f}%")
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc_test:.3f}%")


        result_dump['epoch'].append(epoch)
        result_dump['acc_val'].append(acc_val)
        result_dump['auc_val'].append(auc_val)
        result_dump['loss_val'].append(loss_val)
        result_dump['acc_test'].append(acc_test)
        result_dump['auc_test'].append(auc_test)
        result_dump['loss_test'].append(loss_test)
        result_dump['throughput'].append(thpt)

    if args_aux.eval_all_models:
        df = pd.DataFrame(result_dump)
        df.to_csv(os.path.join(config.OUTPUT, 'summary_test.csv'))
        print(os.path.join(config.OUTPUT, 'summary_test.csv'))

    df_val = pd.DataFrame(stats_val)
    df_val.to_csv(os.path.join(config.OUTPUT, 'stats_val.csv'))

    df_test = pd.DataFrame(stats_test)
    df_test.to_csv(os.path.join(config.OUTPUT, 'stats_test.csv'))


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    roc_auc_meter = AverageMeter()

    stats = {'image_name': [], 'pred': [], 'target': []}
    for c in range(config.MODEL.NUM_CLASSES):
        stats['output_'+str(c)] = []

    end = time.time()
    for idx, (images, target, name) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        pred = output.argmax(dim=1)

        # measure accuracy and record loss
        loss = criterion(output, target)
        if config.DATA.DATASET in ["LVOT", "OCXR"]:
            acc1 = accuracy(output, target, topk=(1, ))[0]
            auc = calculate_roc_auc(output, target)
            # auc = reduce_tensor(auc)
            roc_auc_meter.update(auc.item(), target.size(0))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # Working with imagenet-type datasets with more than 5 classes
        if config.DATA.DATASET not in ["LVOT", "OCXR"]:
            acc5 = reduce_tensor(acc5)
            acc5_meter.update(acc5.item(), target.size(0))

        # Record the image name, pred and target information
        stats['image_name'].extend(name)
        for c in range(config.MODEL.NUM_CLASSES):
            stats['output_'+str(c)].extend(output[:, c].detach().cpu().numpy())
        stats['pred'].extend(pred.detach().cpu().numpy())
        stats['target'].extend(target.detach().cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if config.DATA.DATASET in ["LVOT", "OCXR"]:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Auc@1 {roc_auc_meter.val:.3f} ({roc_auc_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
            else:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
    if config.DATA.DATASET in ["LVOT", "OCXR"]:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f}  * AUC {roc_auc_meter.avg:.3f}')
        return acc1_meter.avg, roc_auc_meter.avg, loss_meter.avg, stats
    else:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return throughput


if __name__ == '__main__':
    _, args_aux, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(),
                           name=f"{config.MODEL.NAME}", obj="test")

    # if dist.get_rank() == 0:
    #     path = os.path.join(config.OUTPUT, "config.json")
    #     with open(path, "w") as f:
    #         f.write(config.dump())
    #     logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())

    main(config, args_aux)