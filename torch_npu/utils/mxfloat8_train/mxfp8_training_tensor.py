import torch
import torch_npu

from torch.distributed._tensor import DTensor
from typing import Dict


aten = torch.ops.aten

# MXFP8：数据用 float8_e4m3fn，scale 是 block-wise 的 E8M0（8-bit 指数，2 的幂）。
# scale 沿最后一维（K 维）按 block_size=32 分组共享。
# npu_dynamic_mx_quant 返回的 shared_exponent 是 uint8，存 E8M0 指数（偏置 127），
# 值 = 2^(e - 127)，传给 npu_quant_matmul 时用 scale_dtype 声明为 e8m0。
MXFP8_BLOCK_SIZE = 32


# @torch._dynamo.allow_in_graph
class _ToMxFP8ConstrFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        if isinstance(input, DTensor):
            input = input.to_local()

        if input.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            input = input.float()

        # 量化数据到 float8_e4m3fn，并生成 block-wise 的 E8M0 scale
        data, scale = torch_npu.npu_dynamic_mx_quant(
            input,
            axis=-1,
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            block_size=MXFP8_BLOCK_SIZE,
        )

        # npu_dynamic_mx_quant 返回的 scale 是 uint8，存 E8M0 指数（偏置 127），
        # 这里不做转换，直接交给 npu_quant_matmul，由 scale_dtype 声明为 e8m0。
        assert scale.dtype == torch.uint8, (
            f"unexpected MXFP8 scale dtype: {scale.dtype}, expected torch.uint8"
        )

        return MxFP8TrainingTensor(
            data=data,
            scale=scale,
            orig_dtype=input.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


# @torch._dynamo.allow_in_graph
class _FromMxFP8ConstrFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        # 反量化：q * scale。scale 是 uint8 存的 E8M0 指数（偏置 127），
        # 可能是 [..., ceilK]（2D）或 [..., ceilK//2, 2]（3D 打包），先展开打包维，
        # 再转成 2^(e-127) 的乘性 scale，沿最后一维每 MXFP8_BLOCK_SIZE 个元素广播。
        out = input._data.float()
        e = input._scale.float()
        if e.dim() >= 3:
            e = e.reshape(*e.shape[:-2], -1)
        scale = torch.where(e == 0, torch.zeros_like(e), torch.pow(2.0, e - 127.0))
        scale = scale.repeat_interleave(MXFP8_BLOCK_SIZE, dim=-1)
        scale = scale[..., : input._data.shape[-1]]
        out = out * scale
        return out.to(input._orig_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class MxFP8TrainingTensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor    # uint8，存 E8M0 指数（偏置 127）
    _orig_dtype: torch.dtype
    __slots__ = ["_data", "_scale", "_orig_dtype"]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype

        return self

    def __repr__(self):
        return (
            f"MxFP8TrainingTensor({self._data}, scale={self._scale}, "
            f"orig_dtype={self._orig_dtype})"
        )

    def __tensor_flatten__(self):
        return ["_data", "_scale"], {"_orig_dtype": self._orig_dtype}

    @staticmethod
    def __tensor_unflatten__(tensor_dict: Dict, metadata, outer_size, outer_stride):
        return MxFP8TrainingTensor(
            tensor_dict["_data"],
            tensor_dict["_scale"],
            metadata["_orig_dtype"],
        )

    def to_original_precision(self):
        return _FromMxFP8ConstrFunc.apply(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        from torch_npu.utils.mxfloat8_train.mxfp8_ops import MXFP8_OPS_TABLE

        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in MXFP8_OPS_TABLE:
            return MXFP8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    __torch_function__ = torch._C._disabled_torch_function_impl


def hp_tensor_to_mxfp8(input: torch.Tensor) -> MxFP8TrainingTensor:
    return _ToMxFP8ConstrFunc.apply(input)
