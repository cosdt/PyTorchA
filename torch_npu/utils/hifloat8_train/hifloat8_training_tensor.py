
import torch
import torch_npu

from torch.distributed._tensor import DTensor
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

# @torch._dynamo.allow_in_graph
class _ToHiFloat8ConstrFunc(torch.autograd.Function):

    @staticmethod
    @torch._dynamo.disable
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        if isinstance(input, DTensor):
            input = input.to_local()
        
        input = input.contiguous().npu().detach()
        if input.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            input = input.float()

        M = input.numel() // input.size(-1)
        input_2d = input.view(M, input.size(-1))

        # Cast data to HIF8 via dynamic quant (real GE op, graph-mode friendly)
        data, scale = torch_npu.npu_dynamic_quant(
            input_2d,
            dst_type=torch_npu.hifloat8,
            dst_type_max=15,
        )
        data = data.view(input.size())

        # Construct HIF8 tensor
        return HiFloat8TrainingTensor(
            data=data,
            scale=scale,
            orig_dtype=input.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


# @torch._dynamo.allow_in_graph
class _FromHiFloat8ConstrFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ):
        data = input._data.contiguous().view(1, -1).detach()
        out = tex.cast_from_fp8(
            data,
            tex.DType.hifloat8,
            NPU_CUSTOM_DType[input._orig_dtype],
        )
        out = out.view(input.size())
        # npu_dynamic_quant 是 scaled quant，反量化乘回 per-token scale（每行一个）
        scale = input._scale.reshape(input.shape[:-1] + (1,))
        out = out * scale.to(out.dtype)
        return out


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class HiFloat8TrainingTensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor    # per-token -> (M,)
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
            f"HiFloat8TrainingTensor({self._data}, scale={self._scale}, "
            f"orig_dtype={self._orig_dtype})"
        )

    def __tensor_flatten__(self):
        return ["_data", "_scale"], {"_orig_dtype": self._orig_dtype}

    @staticmethod
    def __tensor_unflatten__(tensor_dict: Dict, metadata, outer_size, outer_stride):
        return HiFloat8TrainingTensor(
            tensor_dict["_data"],
            tensor_dict["_scale"],
            metadata["_orig_dtype"],
        )

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
