import torch
import torch_npu
from torch import nn

from typing import Callable, Optional

from torch_npu.utils.mxfloat8_train.mxfp8_training_tensor import (
    MXFP8_BLOCK_SIZE,
    MxFP8TrainingTensor,
    hp_tensor_to_mxfp8,
)


def _quant_x1(t: torch.Tensor) -> MxFP8TrainingTensor:
    """沿最后一维分块量化，作为 x1 使用（pertoken_scale 对应其第一维）。"""
    return hp_tensor_to_mxfp8(t)


def _quant_x2(t: torch.Tensor) -> MxFP8TrainingTensor:
    """作为 x2 使用。

    npu_quant_matmul 的 x2 布局为 [reduce_dim, output_dim]，block scale 沿
    reduce 维（x2 的第一维）分组。因此先转置再沿最后一维量化，最后把 data 与
    scale 都转置回来。
    """
    data, scale = torch_npu.npu_dynamic_mx_quant(
        t.t(),
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=MXFP8_BLOCK_SIZE,
    )
    return MxFP8TrainingTensor(data.t(), scale.transpose(0, 1), t.dtype)


@torch._dynamo.allow_in_graph
class matmul_with_mxfp8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        input_shape = input.shape

        input_2d = input.reshape(-1, input.shape[-1])

        input_mxfp8 = _quant_x1(input_2d)            # [M, K], scale=[M, ceilK]
        weight_mxfp8 = hp_tensor_to_mxfp8(weight)    # [N, K], scale=[N, ceilK]

        # 权重数据与 scale 都转置，匹配 npu_quant_matmul 的 x2 布局（[K, N]）
        # scale 是三维（带打包维），用 transpose(0, 1) 交换前两维、保留打包维
        weight_t = MxFP8TrainingTensor(
            weight_mxfp8._data.t(),                  # [K, N]
            weight_mxfp8._scale.transpose(0, 1),     # [ceilK, N, 2]
            weight_mxfp8._orig_dtype,
        )

        # 反向两个 GEMM 的 reduce 维分别是 N、M，与 forward 的 K 不同；
        # block scale 有方向性，不能复用 forward 的量化结果，因此这里预先把
        # weight、input 按各自 reduce 维量化并缓存，反向直接复用。
        input_for_gradw = _quant_x2(input_2d)        # [M, K], scale=[ceilM, K]（沿 M）
        weight_for_gradi = _quant_x2(weight)         # [N, K], scale=[ceilN, K]（沿 N）

        ctx.save_for_backward(input_for_gradw, weight_for_gradi)
        ctx.input_shape = input_shape

        output = torch.mm(input_mxfp8, weight_t)

        output = output.reshape(*input_shape[:-1], -1)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_for_gradw, weight_for_gradi = ctx.saved_tensors

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])  # [M, N]

        # grad_input = grad_output @ weight = [M, N] @ [N, K]
        grad_output_mxfp8 = _quant_x1(grad_output_2d)   # x1=[M, N], scale=[M, ceilN]
        grad_input = torch.mm(grad_output_mxfp8, weight_for_gradi)  # [M, N] @ [N, K] = [M, K]

        # grad_weight = grad_output^T @ input = [N, M] @ [M, K]
        grad_output_t_mxfp8 = _quant_x1(grad_output_2d.t())  # x1=[N, M], scale=[N, ceilM]
        grad_weight = torch.mm(grad_output_t_mxfp8, input_for_gradw)  # [N, M] @ [M, K] = [N, K]

        return grad_input.reshape(ctx.input_shape), grad_weight


class MxFP8Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = matmul_with_mxfp8.apply(input, self.weight)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod: torch.nn.Linear):
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    if isinstance(module, nn.Linear) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return from_float_func(
            module,
        )

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if isinstance(module, nn.Linear) and (
            module_filter_fn is None or module_filter_fn(module, cur_fqn)
        ):
            assert parent_module is not None, (
                f"Linear root module should return early: {module}"
            )
            new_linear_module = from_float_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_linear_module)

    post_order_traversal(root_module)
    return root_module


def convert_to_mxfp8_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:

    torch._C._log_api_usage_once("...")

    def from_float(m: nn.Linear) -> MxFP8Linear:
        return MxFP8Linear.from_float(m)

    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )
