import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
import paddlex as pdx
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_semi, train_one_epoch_burnin
from models import build_model, build_model_semi
from collections import OrderedDict


def get_args_parser():
    parser = argparse.ArgumentParser('Omni-DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=400, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file',
                        default='coco_omni')  # coco_omni, coco_35to80_tagsU, coco_35to80_point, coco_objects_tagsU, coco_objects_points, bees_omni, voc_semi_ voc_omni, objects_omni, crowdhuman_omni
    parser.add_argument('--data_path', default='./coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='results_tmp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--percent', default='10',
                        help='percent with fully labeled')
    parser.add_argument('--BURN_IN_STEP', default=20, type=int,
                        help='as the name means')
    parser.add_argument('--TEACHER_UPDATE_ITER', default=1, type=int,
                        help='as the name means')
    parser.add_argument('--EMA_KEEP_RATE', default=0.9996, type=float,
                        help='as the name means')
    parser.add_argument('--annotation_json_label', default='',
                        help='percent with fully labeled')

    # Consistency arguments
    parser.add_argument('--lamda_c', default=0.1, type=float,
                        help='lamda for consistency loss')
    parser.add_argument('--lamda_s', default=1.0, type=float,
                        help='lamda for consistency loss')
    parser.add_argument('--lamda_t', default=1.0, type=float,
                        help='lamda for consistency loss')
    parser.add_argument('--lamda_dist', default=0.1, type=float,
                        help='lamda for consistency loss')
    parser.add_argument('--T', default=0.5, type=float,
                        help='temperature for consistency loss')
    parser.add_argument('--kd_weight', default=0.1, type=float,
                        help='weight for consistency loss')

    return parser


def main(args):
    pdx.utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(pdx.utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = paddle.device.get_device()
    paddle.seed(args.seed)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_val_semi = build_dataset(image_set='val', args=args, subset='semi')
    print(f"Train: {len(dataset_train)}")
    print(f"Val: {len(dataset_val)}")
    print(f"Val semi: {len(dataset_val_semi)}")

    sampler_train = paddle.io.RandomSampler(dataset_train)
    sampler_val = paddle.io.SequentialSampler(dataset_val)
    sampler_val_semi = paddle.io.SequentialSampler(dataset_val_semi)

    batch_sampler_train = paddle.io.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, collate_fn=utils.collate_fn
    )
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 num_workers=args.num_workers, collate_fn=utils.collate_fn)
    data_loader_val_semi = DataLoader(dataset_val_semi, args.batch_size, sampler=sampler_val_semi, drop_last=False,
                                      num_workers=args.num_workers, collate_fn=utils.collate_fn)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.distributed:
        model = paddle.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = paddle.load(args.resume)
        model.set_state_dict(checkpoint['model'])
        if not args.eval:
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, criterion, data_loader_train, optimizer, device, epoch)
        if args.eval:
            evaluate(args, model, criterion, postprocessors, data_loader_val, base_ds=dataset_val, device=device)

        if args.eval:
            evaluate(args, model, criterion, postprocessors, data_loader_val_semi, base_ds=dataset_val_semi,
                     device=device)

        if args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pdparams')
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint{epoch:04}.pdparams')
            paddle.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'epoch': epoch,
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Omni-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
