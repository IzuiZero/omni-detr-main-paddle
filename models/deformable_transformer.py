import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Normal
from paddle.nn import LayerNorm

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

class DeformableTransformer(nn.Layer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = self.create_parameter((num_feature_levels, d_model), default_initializer=Normal())
        
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                Uniform(-1 / math.sqrt(p.shape[1]), 1 / math.sqrt(p.shape[1]))(p)
        for m in self.sublayers():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            Uniform(-1 / math.sqrt(self.reference_points.weight.shape[0]), 1 / math.sqrt(self.reference_points.weight.shape[0]))(self.reference_points.weight)
            Constant(0.)(self.reference_points.bias)
        Normal()(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = paddle.arange(num_pos_feats, dtype='float32', device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals.unsqueeze(-1) / dim_t
        pos = paddle.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), axis=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].reshape((N_, H_, W_, 1))
            valid_H = paddle.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = paddle.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = paddle.meshgrid(paddle.linspace(0, H_ - 1, H_, dtype='float32', device=memory.device),
                                            paddle.linspace(0, W_ - 1, W_, dtype='float32', device=memory.device))
            grid = paddle.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = paddle.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape((N_, 1, 1, 2))
            grid = (grid.unsqueeze(0).expand((N_, -1, -1, -1)) + 0.5) / scale
            wh = paddle.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = paddle.concat((grid, wh), -1).reshape((N_, -1, 4))
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = paddle.concat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = paddle.log(output_proposals / (1 - output_proposals))
        output_proposals = paddle.masked_fill(output_proposals, memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = paddle.masked_fill(output_proposals, ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = paddle.masked_fill(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = paddle.masked_fill(output_memory, ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = paddle.sum(~mask[:, :, 0], 1)
        valid_W = paddle.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.cast('float32') / H
        valid_ratio_w = valid_W.cast('float32') / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].unsqueeze(0).unsqueeze(0)  # make the position embedding containing level info
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64', device=src_flatten.device)
        level_start_index = paddle.concat((paddle.zeros((1, ), dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = paddle.topk(enc_outputs_class[..., 0], topk, axis=1)[1]
            topk_coords_unact = paddle.gather(enc_outputs_coord_unact, topk_proposals.unsqueeze(-1).tile((1, 1, 4)))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = pos_trans_out.split(c, axis=2)
        else:
            query_embed, tgt = query_embed.split(c, axis=1)
            query_embed = query_embed.unsqueeze(0).tile((bs, 1, 1))
            tgt = tgt.unsqueeze(0).tile((bs, 1, 1))
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None

class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = paddle.meshgrid(paddle.linspace(0.5, H_ - 0.5, H_, dtype='float32', device=device),
                                          paddle.linspace(0.5, W_ - 0.5, W_, dtype='float32', device=device))
            ref_y = ref_y.reshape((-1))[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape((-1))[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = paddle.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerDecoderLayer(nn.Layer):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * paddle.concat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(intermediate_reference_points)

        return output, reference_points

def _get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        two_stage=args.deformable_decoupling,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        num_feature_levels=args.num_feature_levels,
        return_intermediate_dec=True,
    )
