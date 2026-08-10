import torch
from torch import nn

from typing import Optional

from torch_npu.utils.hifloat8_train.hifloat8_training_tensor import hp_tensor_to_hifloat8

@torch._dynamo.allow_in_graph
class matmul_with_hifloat8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        ctx.save_for_backward(input, weight)

        input_hif8 = hp_tensor_to_hifloat8(input)
        weight_hif8 = hp_tensor_to_hifloat8(weight)

        output = torch.mm(input_hif8, weight_hif8)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors

        input_hif8 = hp_tensor_to_hifloat8(input)
        weight_hif8 = hp_tensor_to_hifloat8(weight)

        grad_input = torch.mm(grad_output, weight_hif8.t())
        grad_weight = torch.mm(input_hif8.t(), grad_output)

        return grad_input, grad_weight



class HiFloat8Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = matmul_with_hifloat8.apply(input, self.weight.t())
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
    from_float_func: callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Optional[callable[[nn.Module, str], bool]] = None,
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


def convert_to_hifloat8_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[callable[[nn.Module, str], bool]] = None,
) -> nn.Module:

    torch._C._log_api_usage_once("...")

    def from_float(m: nn.Linear) -> HiFloat8Linear:
        return HiFloat8Linear.from_float(m)

    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )
