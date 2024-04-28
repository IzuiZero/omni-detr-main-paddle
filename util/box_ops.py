# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import paddle
from paddle.vision.ops import box_iou
from paddle.vision.ops import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = paddle.unbind(x, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = paddle.unbind(x, axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)


# modified from torchvision to also return the union
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = paddle.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return paddle.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = paddle.arange(0, h, dtype='float32')
    x = paddle.arange(0, w, dtype='float32')
    y, x = paddle.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(axis=-1)[0]
    x_min = x_mask.masked_fill(~(masks.astype('bool')), 1e8).flatten(1).min(axis=-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(axis=-1)[0]
    y_min = y_mask.masked_fill(~(masks.astype('bool')), 1e8).flatten(1).min(axis=-1)[0]

    return paddle.stack([x_min, y_min, x_max, y_max], axis=1)
