import paddle
import paddle.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Layer):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = paddle.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = paddle.concat([v["labels"] for v in targets])
        tgt_bbox = paddle.concat([v["boxes"] for v in targets])

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = paddle.dist(out_bbox.unsqueeze(1), tgt_bbox.unsqueeze(0), p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(paddle.to_tensor(i, dtype='int64'), paddle.to_tensor(j, dtype='int64')) for i, j in indices]

class HungarianMatcherSemi(nn.Layer):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, indicators=None):
        if indicators is None:
            num_batch = outputs['pred_logits'].shape[0]
            indicators = [1 for i in range(num_batch)]

        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = paddle.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = paddle.concat([v["labels"] for v in targets])
        tgt_bbox = paddle.concat([v["boxes"] for v in targets])

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = paddle.dist(out_bbox.unsqueeze(1), tgt_bbox.unsqueeze(0), p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        indicator_matrix = paddle.concat([paddle.ones((num_queries, cost_class.shape[-1])) * i for i in indicators])
        indicator_matrix = indicator_matrix.cuda()

        C = self.cost_bbox * cost_bbox * indicator_matrix + self.cost_class * cost_class + self.cost_giou * cost_giou * indicator_matrix
        C = C.view(bs, num_queries, -1).cpu()

        if paddle.isinf(C).any():
            C_np = C.numpy()
            np.save('C_isneginf.npy', C_np)
            C[paddle.isinf(C)] = -1e5
        if paddle.isnan(C).any():
            C_np = C.numpy()
            np.save('C_isnan.npy', C_np)
            C[paddle.isnan(C)] = 1e5

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(paddle.to_tensor(i, dtype='int64'), paddle.to_tensor(j, dtype='int64')) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)

def build_matcher_semi(args):
    return HungarianMatcherSemi(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
