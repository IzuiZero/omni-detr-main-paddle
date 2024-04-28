import paddle
import paddle.nn as nn
import math
from util.misc import NestedTensor

class PositionEmbeddingSine(nn.Layer):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = paddle.logical_not(mask)
        y_embed = paddle.cumsum(not_mask, axis=1, dtype='float32')
        x_embed = paddle.cumsum(not_mask, axis=2, dtype='float32')
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype='float32', device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.concat((paddle.sin(pos_x[:, :, :, 0::2]), paddle.cos(pos_x[:, :, :, 1::2])), axis=3).flatten(start_axis=3)
        pos_y = paddle.concat((paddle.sin(pos_y[:, :, :, 0::2]), paddle.cos(pos_y[:, :, :, 1::2])), axis=3).flatten(start_axis=3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))
        return pos


class PositionEmbeddingLearned(nn.Layer):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        self.row_embed.weight.uniform_()
        self.col_embed.weight.uniform_()

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = paddle.arange(w, device=x.device)
        j = paddle.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = paddle.concat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], axis=-1).transpose((2, 0, 1)).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
