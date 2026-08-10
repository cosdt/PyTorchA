"""
测试 HiFloat8 线性层替换和训练。

用法:
    python test.py
"""

import torch
import torch_npu
from torch_npu.utils.hifloat8_train.hifloat8_linear import (
    HiFloat8Linear,
    convert_to_hifloat8_training,
)


def build_simple_model(in_features=64, hidden_features=128, out_features=10):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_features),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_features, out_features),
    )


def test_forward():
    print("=" * 60)
    print("Test 1: Forward pass (same weights, fp vs hif8)")
    print("=" * 60)

    torch.manual_seed(42)

    model = build_simple_model().npu().bfloat16()

    x = torch.randn(4, 64, device='npu', dtype=torch.bfloat16)

    with torch.no_grad():
        fp_out = model(x)

    model = convert_to_hifloat8_training(model)

    with torch.no_grad():
        hif8_out = model(x)

    print(f"FP32   output sample: {fp_out[0, :5]}")
    print(f"HiFloat8 output sample: {hif8_out[0, :5]}")

    cos_sim = torch.nn.functional.cosine_similarity(
        fp_out.flatten().float(), hif8_out.flatten().float(), dim=0
    )
    print(f"Cosine similarity: {cos_sim.item():.6f} (should be close to 1.0)")


def test_backward():
    print("=" * 60)
    print("Test 2: Backward pass")
    print("=" * 60)

    torch.manual_seed(42)

    model = build_simple_model().npu().bfloat16()
    model = convert_to_hifloat8_training(model)

    x = torch.randn(4, 64, device='npu', dtype=torch.bfloat16)
    target = torch.randint(0, 10, (4,), device='npu')

    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.float(), target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}, shape={list(param.shape)}")
        else:
            print(f"  {name}: grad=None")



def test_training_step():
    print("=" * 60)
    print("Test 3: Multi-step training")
    print("=" * 60)

    torch.manual_seed(42)

    model = build_simple_model().npu().bfloat16()
    model = convert_to_hifloat8_training(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(8, 64, device='npu', dtype=torch.bfloat16)
    target = torch.randint(0, 10, (8,), device='npu')

    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.float(), target)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: loss={loss.item():.6f}")

    print()


if __name__ == "__main__":
    test_forward()
    test_backward()
    test_training_step()
    print("All tests passed!")
