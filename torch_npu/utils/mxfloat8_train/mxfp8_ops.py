import torch
import torch_npu

from typing import Any, Dict, Tuple

from torch_npu.utils.mxfloat8_train.mxfp8_training_tensor import (
    MxFP8TrainingTensor,
)
from torch.utils._pytree import tree_map

aten = torch.ops.aten

MXFP8_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the mxfp8 op table"""

    def decorator(func):
        for op in aten_ops:
            if op in MXFP8_OPS_TABLE:
                raise RuntimeError(
                    f"MXFP8 op {op} is already registered to {MXFP8_OPS_TABLE[op].__name__}"
                )
            MXFP8_OPS_TABLE[op] = func
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
def mxfp8_desugar_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return MxFP8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements(
    [
        aten.detach.default,
    ]
)
def mxfp8_desugar_data_and_scale_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return MxFP8TrainingTensor(
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
def mxfp8_transpose(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)

    return MxFP8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements([aten.view.default])
def mxfp8_view(aten_op, args, kwargs=None):

    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return MxFP8TrainingTensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
    )


@implements([aten.split.Tensor])
def mxfp8_split(aten_op, args, kwargs=None):
    new_data_tensors = aten_op(args[0]._data, *args[1:], **kwargs)

    def make_mxfp8(data):
        return MxFP8TrainingTensor(
            data,
            args[0]._scale,
            args[0]._orig_dtype,
        )

    out = map(make_mxfp8, new_data_tensors)
    return list(out)


@implements([aten.cat.default])
def mxfp8_cat(aten_op, args, kwargs=None):
    chunked_tensors: Tuple[MxFP8TrainingTensor, ...] = args[0]

    orig_dtype = chunked_tensors[0]._orig_dtype

    chunk_data = []
    for chunk in chunked_tensors:
        chunk_data.append(chunk._data.view(torch.uint8))

    new_data = aten_op(chunk_data, *args[1:], **kwargs)
    new_data = new_data.view(torch.uint8)
    return MxFP8TrainingTensor(new_data, chunked_tensors[0]._scale, orig_dtype)


@implements([aten.sum.dim_IntList])
def mxfp8_cast_up_op(aten_op, args, kwargs=None):

    def unwrap(x):
        if isinstance(x, MxFP8TrainingTensor):
            return x.to_original_precision()
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


@implements([aten.mm.default, aten.matmul.default])
def mxfp8_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]

    assert isinstance(a, MxFP8TrainingTensor) and isinstance(
        b, MxFP8TrainingTensor
    ), f"Expecting both MxFP8TrainingTensor for mm inputs but found {type(a)} and {type(b)}"

    # MXFP8：scale 为 E8M0，沿 K 维按 32 一组广播
    output = torch_npu.npu_quant_matmul(
        a._data,
        b._data,
        b._scale,
        output_dtype=b._orig_dtype,
        pertoken_scale=a._scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )

    return output


@implements([aten.is_same_size.default])
def mxfp8_is_same_size(aten_op, args, kwargs=None):
    return args[0].shape == args[1].shape
