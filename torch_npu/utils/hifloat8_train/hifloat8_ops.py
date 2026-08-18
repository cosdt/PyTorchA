import torch
import torch_npu

from typing import Any, Dict, Tuple

from torch_npu.utils.hifloat8_train.hifloat8_training_tensor import (
    HiFloat8TrainingTensor,
)
from torch.utils._pytree import tree_map

aten = torch.ops.aten

HIFLOAT8_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the float8 op table"""

    def decorator(func):
        for op in aten_ops:
            if op in HIFLOAT8_OPS_TABLE:
                raise RuntimeError(
                    f"HiFloat8 op {op} is already registered to {HIFLOAT8_OPS_TABLE[op].__name__}"
                )
            HIFLOAT8_OPS_TABLE[op] = func
        return func

    return decorator


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
def hifloat8_desugar_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return HiFloat8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements(
    [
        aten.detach.default,
    ]
)
def hifloat8_desugar_data_and_scale_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return HiFloat8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements(
    [
        aten.t.default,
        aten.transpose.int,
    ]
)
def hifloat8_transpose(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)

    return HiFloat8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements([aten.view.default])
def hifloat8_view(aten_op, args, kwargs=None):

    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return HiFloat8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements([aten.split.Tensor])
def hifloat8_split(aten_op, args, kwargs=None):
    new_data_tensors = aten_op(args[0]._data, *args[1:], **kwargs)

    def make_hifloat8(data):
        return HiFloat8TrainingTensor(
            data,
            args[0]._scale,
            args[0]._orig_dtype,
        )

    out = map(make_hifloat8, new_data_tensors)
    return list(out)


@implements([aten.cat.default])
def hifloat8_cat(aten_op, args, kwargs=None):
    chunked_tensors: Tuple[HiFloat8TrainingTensor, ...] = args[0]

    orig_dtype = chunked_tensors[0]._orig_dtype

    chunk_data = []
    for chunk in chunked_tensors:
        chunk_data.append(chunk._data.view(torch.uint8))

    new_data = aten_op(chunk_data, *args[1:], **kwargs)
    new_data = new_data.view(torch.uint8)
    return HiFloat8TrainingTensor(new_data, chunked_tensors[0]._scale, orig_dtype)


@implements([aten.sum.dim_IntList])
def hifloat8_cast_up_op(aten_op, args, kwargs=None):

    def unwrap(x):
        if isinstance(x, HiFloat8TrainingTensor):
            return x.to_original_precision()
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


@implements([aten.mm.default, aten.matmul.default])
def hifloat8_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]

    assert isinstance(a, HiFloat8TrainingTensor) and isinstance(
        b, HiFloat8TrainingTensor
    ), f"Expecting both HiFloat8TrainingTensor for mm inputs but found {type(a)} and {type(b)}"

    # npu_quant_matmul 约束：
    #   scale：1 维 (t,)，t == 1 或 n，n 为 x2 的最后一维
    #   pertoken_scale：1 维 (m,)，m 为 x1 的倒数第二维
    def normalize_scale(scale, expect, name):
        scale = scale.reshape(-1)
        if scale.numel() == 1:
            return scale
        assert scale.numel() == expect, (
            f"{name} numel {scale.numel()} does not match expected {expect}"
        )
        return scale

    scale = normalize_scale(b._scale, b.shape[-1], "scale")
    pertoken_scale = normalize_scale(a._scale, a.shape[-2], "pertoken_scale")

    output = torch_npu.npu_quant_matmul(
        a._data,
        b._data,
        scale,
        output_dtype=torch.bfloat16,
        pertoken_scale=pertoken_scale,
        x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8,
    )

    return output


@implements([aten.is_same_size.default])
def hifloat8_is_same_size(aten_op, args, kwargs=None):
    return args[0].shape == args[1].shape
