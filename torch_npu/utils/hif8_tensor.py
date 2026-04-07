# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with HIF8 data"""
from __future__ import annotations

__all__ = []

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils._pytree import tree_map
from torch._subclasses.fake_tensor import FakeTensor
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


# init transformer engine
torch_npu._C._cd_init()

tex = torch_npu._C._cd
aten = torch.ops.aten
HIF8_OPS_TABLE: Dict[Any, Any] = {}

NPU_CUSTOM_DType = {
    torch.uint8: tex.DType.uint8,
    torch.int32: tex.DType.int32,
    torch.float32: tex.DType.float32,
    torch.half: tex.DType.float16,
    torch.bfloat16: tex.DType.bfloat16,
}

def implements(aten_ops):
    """Register aten ops to the HIF8 op table."""

    def decorator(func):
        for op in aten_ops:
            if op in HIF8_OPS_TABLE:
                raise RuntimeError(
                    f"HIF8 op {op} is already registered to {HIF8_OPS_TABLE[op].__name__}"
                )
            HIF8_OPS_TABLE[op] = func
        return func

    return decorator

def _is_fakeish_tensor(x):
    return (
        isinstance(x, torch.Tensor)
        and (
            isinstance(x, FakeTensor)
            or x.device.type == "meta"
            or type(x).__name__ == "FunctionalTensor"
        )
    )

@torch._dynamo.allow_in_graph
class _FromHiFloat8Func(torch.autograd.Function):
    """Cast from HIF8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: _HiFloat8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor.dtype
        data = tensor._data.contiguous().view(1, -1).detach()
        out = tex.cast_from_fp8(
            data,
            NPU_CUSTOM_DType[dtype], # tex.DType.hifloat8,
            NPU_CUSTOM_DType[dtype],
        )
        out = out.view(tensor.size())
        return out

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # Assume that we want gradients in full precision
        return grad, None


@torch._dynamo.allow_in_graph
class _ToHiFloat8Func(torch.autograd.Function):
    """Cast to HIF8 from other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: torch.Tensor,
    ) -> _HiFloat8Tensor:

        # Check input tensor TODO
        tensor = tensor.contiguous().npu().detach()
        if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            tensor = tensor.float()

        # Cast data to HIF8
        data = tex.cast_to_fp8(
            tensor.view(1, -1),
            NPU_CUSTOM_DType[tensor.dtype], # tex.DType.hifloat8,
        )
        data = data.view(tensor.size())

        # Construct HIF8 tensor
        return _HiFloat8Tensor(
            data=data,
            dtype=tensor.dtype,
        )

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # Assume that we want gradients in full precision
        return grad, None

def _from_hifloat8_impl(
    tensor: _HiFloat8Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = tensor.dtype
    data = tensor._data
    if _is_fakeish_tensor(data):
        if tuple(data.size()) != tuple(tensor.size()):
            data = data.view(tensor.size())
        if data.dtype != dtype:
            data = data.to(dtype)
        return data
    return _FromHiFloat8Func.apply(tensor, dtype)


def _to_hifloat8_impl(tensor: torch.Tensor) -> _HiFloat8Tensor:
    if _is_fakeish_tensor(tensor):
        return _HiFloat8Tensor(
            data=tensor,
            dtype=tensor.dtype,
        )
    return _ToHiFloat8Func.apply(tensor)


class _HiFloat8Tensor(torch.Tensor):
    """Experimental tensor class with HIF8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) HIF8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw HIF8 data in a uint8 tensor
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.

    """

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        # Check that data buffer is valid
        # if data.element_size() != 1:
        #     raise ValueError(
        #         f"HiFloat8Tensor requires data buffer with 8-bit dtype (got dtype={data.dtype})"
        #         + pta_error(ErrCode.VALUE)
        #     )
        if data.requires_grad:
            raise ValueError(
                "HiFloat8Tensor requires non-differentiable data buffer"
                + pta_error(ErrCode.VALUE)
            )
        if not _is_fakeish_tensor(data) and not data.is_npu:
            data = data.npu()

        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data

        return self

    @classmethod
    def make_like(
        cls,
        tensor: _HiFloat8Tensor,
        *,
        data: torch.Tensor,
        **kwargs,
    ) -> _HiFloat8Tensor:
        """Use attributes of a _HiFloat8Tensor to create another _HiFloat8Tensor

        See constructor for list of keyword arguments.

        """
        default_kwargs = dict(
            dtype=tensor.dtype,
        )
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        return _HiFloat8Tensor(data=data, **kwargs)

    def __repr__(self):
        return (
            "HiFloat8Tensor("
            f"data={self.from_hifloat8(dtype=self.dtype)}"
            ")"
        )

    def from_hifloat8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct PyTorch tensor from _HiFloat8Tensor

        By default the resulting tensor's dtype is the
        _HiFloat8Tensor's nominal dtype.
        """
        return _from_hifloat8_impl(self, dtype)

    @classmethod
    def to_hifloat8(
        cls,
        tensor: torch.Tensor
    ):
        """Construct _HiFloat8Tensor from PyTorch tensor"""
        return _to_hifloat8_impl(tensor)

    def float(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return self.from_hifloat8().cpu()

    def clone(self) -> _HiFloat8Tensor:
        return aten.clone.default(self)

    def view(self, *shape: Tuple[int]) -> _HiFloat8Tensor:
        return aten.view.default(self, shape)

    def reshape(self, *shape: Tuple[int]) -> _HiFloat8Tensor:
        return aten.reshape.default(self, shape)

    def transpose(self, dim0, dim1):
        return aten.transpose.int(self, dim0, dim1)

    def contiguous(
        self,
        *,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> _HiFloat8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data.is_contiguous(memory_format=memory_format):
            return self
        return _HiFloat8Tensor.make_like(
            self,
            data=self._data.detach().contiguous(memory_format=memory_format),
        )

    def to_dtype(self, dtype: torch.dtype) -> _HiFloat8Tensor:
        """Create `_HiFloat8Tensor` with given nominal dtype

        The new tensor has the same underlying HIF8 data.

        """
        return _HiFloat8Tensor.make_like(
            self,
            data=self._data,
            dtype=dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        def allowed_subclasses(type_):
            return (
                issubclass(cls, type_)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type_)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type_
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in HIF8_OPS_TABLE:
            return HIF8_OPS_TABLE[func](func, args, kwargs)
        
        def maybe_unwrap(t):
            if isinstance(t, _HiFloat8Tensor):
                return _from_hifloat8_impl(t)
            return t

        def maybe_update_inplace(arg, new_arg, schema_arg):
            """Update values of HIF8 tensors

            Keep the same HIF8 scaling factors.

            """
            check_args = isinstance(arg, _HiFloat8Tensor) and isinstance(new_arg, torch.Tensor)
            check_schema = (
                hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            )

            if check_args and check_schema:
                arg.copy_(new_arg)

        # In-place op
        if func._schema.is_mutable:
            # Cast to higher precision, perform op, and cast values
            # back to original HIF8 buffers
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                if not (kwarg == new_kwarg == schema_arg.name):
                    raise ValueError('name of the kw argument should match' + pta_error(ErrCode.VALUE))
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op
        # Note: cast to higher precision and perform op
        args = tree_map(maybe_unwrap, args)
        kwargs = tree_map(maybe_unwrap, kwargs)
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        dtype: torch.dtype,
    ) -> _HiFloat8Tensor:
        """Build _HiFloat8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return _HiFloat8Tensor(
            data=data,
            dtype=dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to HIF8 metadata objects"""
        return (
            _HiFloat8Tensor._make_in_reduce_ex,
            (self._data, self.dtype),
        )

    def _get_data(self) -> _HiFloat8Tensor:
        """Get tensor data property"""
        return super().data

    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Cast tensor to HIF8 and store in HIF8 buffer.

        """
        with torch.no_grad():
            self.copy_(tensor)

    # Cast to HIF8 when setting _HiFloat8Tensor.data
    data = property(_get_data, _set_data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def __tensor_flatten__(self):
        return ["_data"], {"dtype": self.dtype}

    @staticmethod
    def __tensor_unflatten__(tensor_data_dict, meta, outer_size, outer_stride):
        data = tensor_data_dict["_data"]
        if outer_size is not None and outer_stride is not None:
            if tuple(data.size()) != tuple(outer_size) or tuple(data.stride()) != tuple(
                outer_stride
            ):
                data = data.as_strided(outer_size, outer_stride, data.storage_offset())


        if isinstance(data, FakeTensor) or data.device.type == "meta":
            shape = outer_size if outer_size is not None else data.shape
            stride = outer_stride if outer_stride is not None else data.stride()
            return torch.empty_strided(
                shape,
                stride,
                dtype=meta["dtype"],
                device=data.device,
            )

        return _HiFloat8Tensor(
            data=data,
            dtype=meta["dtype"],
        )
    
def _wrap_hif8_like(
    tensor: _HiFloat8Tensor,
    data: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> _HiFloat8Tensor:
    return _HiFloat8Tensor.make_like(
        tensor,
        data=data,
        dtype=tensor.dtype if dtype is None else dtype,
    )


def _unwrap_hif8(x):
    if isinstance(x, _HiFloat8Tensor):
        return _from_hifloat8_impl(x)
    return x

@implements(
    [
        aten._unsafe_view.default,
        aten.as_strided.default,
        aten.clone.default,
        aten.slice.Tensor,
        aten.fill_.Scalar,
        aten.reshape.default,
    ]
)
def hif8_desugar_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **(kwargs or {}))
    return _wrap_hif8_like(args[0], new_data)


@implements([aten.detach.default])
def hif8_detach(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **(kwargs or {}))
    return _wrap_hif8_like(args[0], new_data)


@implements([aten.t.default, aten.transpose.int])
def hif8_transpose(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **(kwargs or {}))
    return _wrap_hif8_like(args[0], new_data)


@implements([aten.view.default])
def hif8_view(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **(kwargs or {}))
    return _wrap_hif8_like(args[0], new_data)


@implements([aten._to_copy.default])
def hif8_to_copy(aten_op, args, kwargs=None):
    kwargs = kwargs or {}
    tensor = args[0]
    target_dtype = kwargs.get("dtype", tensor.dtype)
    data_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
    new_data = tensor._data
    if data_kwargs:
        new_data = aten_op(new_data, **data_kwargs)
    return _wrap_hif8_like(tensor, new_data, dtype=target_dtype)


@implements([aten.mm.default, aten.matmul.default])
def hif8_matmul(aten_op, args, kwargs=None):
    kwargs = kwargs or {}
    new_args = tree_map(_unwrap_hif8, args)
    new_kwargs = tree_map(_unwrap_hif8, kwargs)
    return aten_op(*new_args, **new_kwargs)


@implements([aten.copy_.default])
def hif8_copy(aten_op, args, kwargs=None):
    kwargs = kwargs or {}
    dst = args[0]
    src = args[1]
    if not isinstance(dst, torch.Tensor):
        raise RuntimeError(
            "Attempted to copy into something that isn't a PyTorch tensor"
            + pta_error(ErrCode.TYPE)
        )
    if not isinstance(src, torch.Tensor):
        raise RuntimeError(
            "Attempted to copy from something that isn't a PyTorch tensor"
            + pta_error(ErrCode.TYPE)
        )

    dst_is_hif8 = isinstance(dst, _HiFloat8Tensor)
    src_is_hif8 = isinstance(src, _HiFloat8Tensor)

    if not dst_is_hif8 and src_is_hif8:
        src_hp = src.from_hifloat8()
        return aten_op(dst, src_hp, *args[2:], **kwargs)

    if dst_is_hif8 and src_is_hif8:
        fp8_out = aten_op(dst._data, src._data, *args[2:], **kwargs)
        return _wrap_hif8_like(dst, fp8_out)

    if dst_is_hif8 and not src_is_hif8:

        if _is_fakeish_tensor(dst._data) or _is_fakeish_tensor(src):
            if not _is_fakeish_tensor(src):
                src = dst._data.new_empty_strided(
                    src.size(),
                    src.stride(),
                    dtype=src.dtype,
                )

            # keep broadcast semantics check
            _ = src.expand(dst.size())

            # copy_ mutates dst; result keeps dst metadata
            return _wrap_hif8_like(dst, dst._data)


        src = src.expand(dst.size())
        src = src.to(
            device=dst.device,
            memory_format=torch.contiguous_format,
        )
        
        if not dst._data.is_contiguous():
            raise RuntimeError(
                "Transformer Engine cast kernels require contiguous data"
                + pta_error(ErrCode.INTERNAL)
            )
        tex.cast_to_fp8_noalloc(
            src.contiguous().view(1, -1),
            dst._data.view(1, -1),
            NPU_CUSTOM_DType[dst._data.dtype],
        )
        return dst

    raise RuntimeError(
        "Using HiFloat8Tensor copy logic, but no HiFloat8Tensor found"
        + pta_error(ErrCode.INTERNAL)
    )