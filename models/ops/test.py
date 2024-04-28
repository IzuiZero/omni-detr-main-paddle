import time
import paddle
import numpy as np

from paddle.autograd import grad
from paddle.autograd import PyLayer

from functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype='int64').cuda()
level_start_index = paddle.concat([paddle.zeros((1, ), dtype='int64'), paddle.cumsum(shapes.prod(-1))[:-1]], axis=0)
S = sum([(H*W).item() for H, W in shapes])

paddle.seed(3)

class MSDeformAttnFunction(PyLayer):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            ms_deform_attn_core_pytorch(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def check_forward_equal_with_pytorch_double():
    value = paddle.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = paddle.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = paddle.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).cpu()
    output_paddle = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).cpu()
    fwdok = paddle.allclose(output_paddle, output_pytorch)
    max_abs_err = (output_paddle - output_pytorch).abs().max()
    max_rel_err = ((output_paddle - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok.numpy()[0]} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err.numpy()[0]:.2e} max_rel_err {max_rel_err.numpy()[0]:.2e}')


def check_forward_equal_with_pytorch_float():
    value = paddle.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = paddle.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = paddle.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).cpu()
    output_paddle = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).cpu()
    fwdok = paddle.allclose(output_paddle, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_paddle - output_pytorch).abs().max()
    max_rel_err = ((output_paddle - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok.numpy()[0]} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err.numpy()[0]:.2e} max_rel_err {max_rel_err.numpy()[0]:.2e}')


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = paddle.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = paddle.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = paddle.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2

    value.stop_gradient = not grad_value
    sampling_locations.stop_gradient = not grad_sampling_loc
    attention_weights.stop_gradient = not grad_attn_weight

    func = MSDeformAttnFunction.apply

    gradok = paddle.autograd.gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok.numpy()[0]} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()

    for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
        check_gradient_numerical(channels, True, True, True)
