# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import paddle
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
import paddle.nn as nn


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (ow, oh)


def box_calibration(cur_boxes, cur_labels, cur_idx, records_unlabel_q, records_unlabel_k, pixels):
    # cur_boxes, num * [x, y, w, h]
    if pixels == 600:
        max_pixels = 1000
    elif pixels == 800:
        max_pixels = 1333

    records_unlabel_q = records_unlabel_q[0]
    records_unlabel_k = records_unlabel_k[0]

    # we first recover the bbox coordinate for weak aug in the original image width height space
    cur_one_tensor = paddle.to_tensor([1, 0, 0, 0])
    cur_one_tensor = cur_one_tensor.cuda()
    cur_one_tensor = cur_one_tensor.repeat(cur_boxes.shape[0], 1)
    if 'RandomFlip' in records_unlabel_k.keys() and records_unlabel_k['RandomFlip']:
        original_boxes = paddle.abs(cur_one_tensor - cur_boxes)
    else:
        original_boxes = cur_boxes
    original_boxes = box_cxcywh_to_xyxy(original_boxes)
    img_w = records_unlabel_k['OriginalImageSize'][1]
    img_h = records_unlabel_k['OriginalImageSize'][0]
    scale_fct = paddle.to_tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.cuda()
    scale_fct = scale_fct.repeat(cur_boxes.shape[0], 1)
    original_boxes = original_boxes * scale_fct
    cur_boxes = paddle.clone(original_boxes)

    # then, we repeat the boxes generation process in the strong aug for the predicted boxes
    if 'RandomFlip' in records_unlabel_q.keys() and records_unlabel_q['RandomFlip']:
        cur_boxes = cur_boxes[:, [2, 1, 0, 3]] * paddle.to_tensor([-1, 1, -1, 1]).cuda() + paddle.to_tensor([img_w, 0, img_w, 0]).cuda()

    if records_unlabel_q['RandomResize_times'] > 1:

        # random resize
        rescaled_size1 = records_unlabel_q['RandomResize_scale'][0]
        rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size1[0]
        img_h = rescaled_size1[1]

        # random size crop
        region = records_unlabel_q['RandomSizeCrop']
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]
        max_size = paddle.to_tensor([w, h], dtype='float32').cuda()
        cropped_boxes = cur_boxes - paddle.to_tensor([j, i, j, i]).cuda()
        cropped_boxes = paddle.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clip(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
        cur_boxes = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")
        cropped_boxes = paddle.clone(cur_boxes)
        cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
        keep = paddle.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]

        img_w = w
        img_h = h

        # random resize
        rescaled_size2 = records_unlabel_q['RandomResize_scale'][1]
        rescaled_size2 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size2, max_size=max_pixels)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size2, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size2[0]
        img_h = rescaled_size2[1]
    else:
        # random resize
        rescaled_size1 = records_unlabel_q['RandomResize_scale'][0]
        rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1, max_size=max_pixels)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size1[0]
        img_h = rescaled_size1[1]

    # finally, deal with normalize part in deformable detr aug code
    cur_boxes = box_xyxy_to_cxcywh(cur_boxes)
    cur_boxes = cur_boxes / paddle.to_tensor([img_w, img_h, img_w, img_h], dtype='float32').cuda()

    # deal with the randomerasing part
    if 'RandomErasing1' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing1']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]
    if 'RandomErasing2' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing2']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]
    if 'RandomErasing3' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing3']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]

    updated_boxes = cur_boxes
    updated_labels = cur_labels
    assert updated_boxes.shape[0] == updated_labels.shape[0]
    return updated_boxes, updated_labels, cur_idx


def unified_filter_pseudo_labels(pseudo_unsup_outputs, targets_unlabel_q, targets_unlabel_k, records_unlabel_q, records_unlabel_k, pixels, label_type, w_p=0.5, w_t=0.5, is_binary=False, threshold=0.7):
    pseudo_unsup_logits = pseudo_unsup_outputs['pred_logits']
    softmax = nn.Softmax(axis=2)
    pseudo_unsup_prob = softmax(pseudo_unsup_logits)

    if is_binary:
        satisfied_idx_d2 = (pseudo_unsup_prob[0, :, 0] > threshold).nonzero(as_tuple=True)
        satisfied_idx_d1 = paddle.zeros(satisfied_idx_d2[0].shape[0], dtype=paddle.int64).cuda()
        satisfied_idx = (satisfied_idx_d1, satisfied_idx_d2[0])
        pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
        satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        satisfied_class = paddle.zeros(satisfied_bbox.shape[0], dtype=paddle.int64).cuda()
    else:
        if label_type == 'Unsup':
            classes_scores, classes_indices = paddle.max(pseudo_unsup_prob, axis=2)
            satisfied_idx = (classes_scores > threshold).nonzero(as_tuple=True)
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
            satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
        elif label_type == 'tagsU':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']
            targets_gt_unlabel_q = paddle.unique(targets_gt_unlabel_q)

            # the first thing we do is predicting count number for each class
            # get the class: object number
            targets_gt_unlabel_q_list = targets_gt_unlabel_q.tolist()
            targets_gt_unlabel_q_new = paddle.zeros(1, dtype=paddle.int64).cuda()
            for i_c in targets_gt_unlabel_q_list:
                classes_scores_i = pseudo_unsup_prob[0, :, i_c]
                satisfied_idx_i = (classes_scores_i > threshold).nonzero(as_tuple=True)
                if len(satisfied_idx_i[0]) > 0:
                    targets_gt_unlabel_q_new = paddle.concat(
                        (targets_gt_unlabel_q_new, i_c * paddle.ones(len(satisfied_idx_i[0]), dtype=paddle.int64).cuda()))
                else:
                    targets_gt_unlabel_q_new = paddle.concat(
                        (targets_gt_unlabel_q_new, i_c * paddle.ones(1, dtype=paddle.int64).cuda()))

            targets_gt_unlabel_q = targets_gt_unlabel_q_new[1:]

            # Compute the distance cost between predictions and tags, we use 1 - confidence score as the distance measurement
            dist_bbox_tags = paddle.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()
            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]
            dist_bbox_tags = dist_bbox_tags.numpy()

            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            indices = scipy.optimize.linear_sum_assignment(dist_bbox_tags)
            updated_idx_d1 = paddle.zeros(targets_gt_unlabel_q.shape[0], dtype=paddle.int64).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = paddle.to_tensor(updated_idx_d2, dtype=paddle.int64).cuda()
            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = targets_gt_unlabel_q
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        elif label_type == 'tagsK':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']

            dist_bbox_tags = paddle.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()

            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]
            dist_bbox_tags = dist_bbox_tags.numpy()

            indices = scipy.optimize.linear_sum_assignment(dist_bbox_tags)

            updated_idx_d1 = paddle.zeros(targets_gt_unlabel_q.shape[0], dtype=paddle.int64).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = paddle.to_tensor(updated_idx_d2, dtype=paddle.int64).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = targets_gt_unlabel_q
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        elif label_type == 'pointsU' or label_type == 'pointsK':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']

            # Compute the distance cost between predictions and tags, we use 1 - confidence score as the distance measurement
            dist_bbox_tags = paddle.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()

            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]

            dist_bbox_tags = dist_bbox_tags.numpy()

            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            targets_gt_point_unlabel_q = targets_unlabel_k[0]['points']

            classes_scores, classes_indices = paddle.max(pseudo_unsup_prob, axis=2)
            matrix_w_high_score = classes_scores[0]
            matrix_w_high_score = paddle.unsqueeze(matrix_w_high_score, 1)
            matrix_w_high_score = 1 - matrix_w_high_score

            # to get an indicator matrix of which prediction a ground truth is included
            indicator_include_matrix = paddle.zeros((pseudo_unsup_prob.shape[1], targets_gt_point_unlabel_q.shape[0]),
                                                   dtype=paddle.int64).cuda()
            for i in range(targets_gt_point_unlabel_q.shape[0]):
                point_i = targets_gt_point_unlabel_q[i, :2]
                x0 = point_i[0]
                y0 = point_i[1]
                keep_i = (pseudo_unsup_bbox[0, :, 0] - pseudo_unsup_bbox[0, :, 2] / 2 < x0) & (
                            pseudo_unsup_bbox[0, :, 0] + pseudo_unsup_bbox[0, :, 2] / 2 > x0) & (
                                     pseudo_unsup_bbox[0, :, 1] - pseudo_unsup_bbox[0, :, 3] / 2 < y0) & (
                                     pseudo_unsup_bbox[0, :, 1] + pseudo_unsup_bbox[0, :, 3] / 2 > y0)
                keep_i = keep_i.int()
                keep_i = keep_i.type(paddle.cuda.Int64Tensor)
                if keep_i.sum() > 0:
                    indicator_include_matrix[keep_i == 1, i] = 1
            indicator_include_matrix[indicator_include_matrix < 1] = 1e3  # just give a random large number

            # Compute the distance cost between boxes
            dist_bbox_point = paddle.cdist(pseudo_unsup_bbox[0][:, :2], targets_gt_point_unlabel_q[:, :2], p=2)

            if dist_bbox_point.shape[1] > 0:
                dist_bbox_point = dist_bbox_point - dist_bbox_point.min()
                dist_bbox_point = dist_bbox_point / paddle.max(dist_bbox_point)
            dist_bbox_point = (dist_bbox_point + matrix_w_high_score) * indicator_include_matrix
            dist_bbox_point = dist_bbox_point.numpy()
            dist_bbox_point = w_p * dist_bbox_point + w_t * dist_bbox_tags

            indices = scipy.optimize.linear_sum_assignment(dist_bbox_point)
            updated_idx_d1 = paddle.zeros(targets_gt_unlabel_q.shape[0], dtype=paddle.int64).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = paddle.to_tensor(updated_idx_d2, dtype=paddle.int64).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            if w_t == 0:
                satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
            else:
                satisfied_class = targets_gt_unlabel_q
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
            satisfied_points = targets_gt_point_unlabel_q
        else:  # boxesEC or boxesU
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            targets_gt_box_unlabel_q = targets_unlabel_k[0]['boxes']

            classes_scores, classes_indices = paddle.max(pseudo_unsup_prob, axis=2)
            matrix_w_high_score = classes_scores[0]
            matrix_w_high_score = paddle.unsqueeze(matrix_w_high_score, 1)
            matrix_w_high_score = 1 - matrix_w_high_score

            # Compute the distance cost between boxes
            # Compute the L1 cost between boxes
            cost_bbox = paddle.cdist(pseudo_unsup_bbox[0], targets_gt_box_unlabel_q, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = paddle.cdist(pseudo_unsup_bbox[0], targets_gt_box_unlabel_q, p=2)

            if cost_giou.shape[1] > 0:
                cost_giou = 1 - cost_giou / (paddle.cdist(pseudo_unsup_bbox[0], paddle.unsqueeze(paddle.to_tensor([0, 0, 1, 1], dtype='float32').cuda(), 0), p=2) + paddle.cdist(targets_gt_box_unlabel_q, paddle.unsqueeze(paddle.to_tensor([0, 0, 1, 1], dtype='float32').cuda(), 0), p=2) - cost_giou)
            else:
                cost_giou = paddle.zeros_like(cost_giou)

            if cost_bbox.shape[1] > 0:
                cost_bbox = cost_bbox - cost_bbox.min()
                cost_bbox = cost_bbox / paddle.max(cost_bbox)
            cost_bbox = (cost_bbox + matrix_w_high_score) * 1e3

            # Compute the giou cost betwen boxes
            cost_giou = (cost_giou + matrix_w_high_score) * 1e3

            dist_bbox_boxes = w_p * cost_bbox + w_t * cost_giou

            indices = scipy.optimize.linear_sum_assignment(dist_bbox_boxes)
            updated_idx_d1 = paddle.zeros(pseudo_unsup_prob.shape[1], dtype=paddle.int64).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = paddle.to_tensor(updated_idx_d2, dtype=paddle.int64).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]

    targets_unlabel_q_new = {'boxes': satisfied_bbox,
                             'labels': satisfied_class,
                             'idx': paddle.arange(0, satisfied_class.shape[0], dtype=paddle.int64).cuda()}
    targets_unlabel_q = [targets_unlabel_q_new]
    return targets_unlabel_q
