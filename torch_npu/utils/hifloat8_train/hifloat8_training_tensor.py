
import torch
import torch_npu

from torch.distributed._tensor import DTensor
from torch.library import custom_op

from typing import Dict


tex = torch_npu._C._cd
aten = torch.ops.aten

NPU_CUSTOM_DType = {
    torch.uint8: tex.DType.uint8,
    torch.int32: tex.DType.int32,
    torch.float32: tex.DType.float32,
    torch.half: tex.DType.float16,
    torch.bfloat16: tex.DType.bfloat16,
}

TEX_DTYPE_TO_TORCH = {v: k for k, v in NPU_CUSTOM_DType.items()}


@custom_op("npu::_hif8_cast_to", mutates_args=())
def _hif8_cast_to(x: torch.Tensor, otype: int) -> torch.Tensor:
    """Cast to HiFloat8 (FP8). ``otype`` 是 c10_npu DType 的取值。"""
    return tex.cast_to_fp8(x, otype)


@_hif8_cast_to.register_fake
def _hif8_cast_to_fake(x, otype):
    return torch.empty(x.shape, dtype=torch.uint8, device=x.device)


@custom_op("npu::_hif8_cast_from", mutates_args=())
def _hif8_cast_from(x: torch.Tensor, itype: int, otype: int) -> torch.Tensor:
    return tex.cast_from_fp8(x, itype, otype)


@_hif8_cast_from.register_fake
def _hif8_cast_from_fake(x, itype, otype):
    dtype = TEX_DTYPE_TO_TORCH.get(otype, torch.float32)
    return torch.empty(x.shape, dtype=dtype, device=x.device)

@torch._dynamo.allow_in_graph
class _ToHiFloat8ConstrFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        input = input.contiguous().npu().detach()
        if input.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            input = input.float()

        # Cast data to HIF8
        data = _hif8_cast_to(
            input.view(1, -1),
            tex.DType.hifloat8,
        )
        data = data.view(input.size())

        # Construct HIF8 tensor
        return HiFloat8TrainingTensor(
            data=data,
            orig_dtype=input.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


@torch._dynamo.allow_in_graph
class _FromHiFloat8ConstrFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        data = input._data.contiguous().view(1, -1).detach()
        out = _hif8_cast_from(
            data,
            tex.DType.hifloat8,
            NPU_CUSTOM_DType[input._orig_dtype],
        )
        out = out.view(input.size())
        return out


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class HiFloat8TrainingTensor(torch.Tensor):
    _data: torch.Tensor
    _orig_dtype: torch.dtype
    __slots__ = ["_data", "_orig_dtype"]

    def __new__(
        cls,
        data: torch.Tensor,
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
        self._orig_dtype = orig_dtype

        return self

    def __repr__(self):
        return f"HiFloat8TrainingTensor({self._data}, orig_dtype={self._orig_dtype})"

    def __tensor_flatten__(self):
        return ["_data"], {"_orig_dtype": self._orig_dtype}

    @staticmethod
    def __tensor_unflatten__(tensor_dict: Dict, metadata, outer_size, outer_stride):
        return HiFloat8TrainingTensor(tensor_dict["_data"], metadata["_orig_dtype"])

    def to_original_precision(self):
        return _FromHiFloat8ConstrFunc.apply(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        from torch_npu.utils.hifloat8_train.hifloat8_ops import HIFLOAT8_OPS_TABLE

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

        if func in HIFLOAT8_OPS_TABLE:
            return HIFLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    __torch_function__ = torch._C._disabled_torch_function_impl


def hp_tensor_to_hifloat8(input: torch.Tensor) -> HiFloat8TrainingTensor:
    return _ToHiFloat8ConstrFunc.apply(input)
