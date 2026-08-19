# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HiFloat8 + DeepSpeed 训练集成测试：Qwen3-8B 微调。
支持 普通 / 分布式 / 图模式 / 分布式+图模式 四种训练模式。

用法:
    # 1. 普通训练（单卡，无图模式）
    deepspeed --num_gpus=1 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode plain

    # 2. 分布式训练（多卡）
    deepspeed --num_gpus=2 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode distributed

    # 3. 图模式训练（单卡 + torch.compile）
    deepspeed --num_gpus=1 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode compile

    # 4. 分布式 + 图模式训练
    deepspeed --num_gpus=2 tests/unit/hifloat8/test_hifloat8_train_modes.py --mode distributed_compile

"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import deepspeed
import deepspeed.comm as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.hifloat8 import is_hifloat8_available

os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

MODEL_PATH = "/home/c30076943/models/Qwen3-8B"
MODEL_DTYPE = torch.bfloat16
DS_COMPILE_BACKEND = os.getenv("DS_COMPILE_BACKEND", "npu")


from pathlib import Path

dataset_path = "/home/c30076943/dataset/wikitext-2-raw/wiki.train.raw"

with open(dataset_path, "r", encoding="utf-8") as f:
    datasets = f.read()


def build_dataset(tokenizer, seq_len):
    encodings = tokenizer(
        datasets,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()
    # 因果 LM 训练时忽略 padding 位置（置 -100，不参与 loss）
    labels[input_ids == tokenizer.pad_token_id] = -100
    return input_ids, labels


def get_ds_config(world_size, micro_batch_size, gradient_accumulation_steps, lr):
    return {
        # 全局 batch = 每卡 micro batch * 梯度累积步数 * 卡数
        "train_batch_size": micro_batch_size * gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": True},
        # "hifloat8": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr},
        },
        "zero_optimization": {
            "stage": 3,  # 使用 ZeRO-3，切分模型参数
            "offload_optimizer": {
                "device": "cpu",  # 优化器状态 offload 到 CPU
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",  # 参数 offload 到 CPU
                "pin_memory": True
            },
            "contiguous_gradients": True,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
    }


def maybe_compile_module(module: nn.Module) -> nn.Module:
    """图模式：在 HiFloat8 层替换之后编译模型，失败时回退到 eager。"""
    if not hasattr(torch, "compile"):
        print("[COMPILE] torch.compile unavailable, skip.", flush=True)
        return module
    try:
        compiled_module = torch.compile(module, backend=DS_COMPILE_BACKEND)
        print(f"[COMPILE] torch.compile enabled, backend={DS_COMPILE_BACKEND}.", flush=True)
        return compiled_module
    except Exception as error:
        print(f"[COMPILE] skip compile, reason: {error}", flush=True)
        return module


def run_training(model_engine, input_ids, labels, steps, log_interval=10):
    micro_batch = model_engine.train_micro_batch_size_per_gpu()
    num_samples = input_ids.shape[0]
    device = "npu"

    for step in range(steps):
        # 每个 rank 用自己的随机顺序取 micro-batch，模拟真实数据并行
        order = torch.randperm(num_samples)[:micro_batch]
        batch_ids = input_ids[order].to(device)
        batch_labels = labels[order].to(device)

        model_engine.zero_grad()
        out = model_engine(input_ids=batch_ids, labels=batch_labels)
        loss = out.loss
        model_engine.backward(loss)
        model_engine.step()

        # if step % log_interval == 0 and dist.get_rank() == 0:
        if dist.get_rank() == 0:
            print(f"Step {step:3d} | loss = {loss.item():.6f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="HiFloat8 + DeepSpeed training modes")
    # DeepSpeed 自己的命令行参数（--deepspeed / --deepspeed_config 等）
    deepspeed.add_config_arguments(parser)
    # deepspeed launcher 默认会注入 --local_rank=<n>，必须接收，否则 argparse 报 unrecognized
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="deepspeed launcher 注入的本地 rank（分布式训练必需）",
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "distributed", "compile", "distributed_compile"],
        default="plain",
        help="plain / distributed / compile / distributed_compile",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="预训练模型路径（默认 Qwen3-0.6B）",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="每卡 micro batch size（默认 1）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="训练序列长度（默认 128）",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（默认 1）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="训练步数（默认 200）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="学习率（默认 1e-5）",
    )

    return parser.parse_args()


def validate_mode(mode, world_size):
    is_distributed_mode = mode in ("distributed", "distributed_compile")
    if is_distributed_mode and world_size == 1:
        raise RuntimeError(
            f"mode={mode} requires multiple NPUs, but world_size={world_size}. "
            "Launch with `deepspeed --num_gpus=N ...` (N >= 2)."
        )
    if not is_distributed_mode and world_size > 1:
        print(f"[WARN] mode={mode} launched with world_size={world_size}, "
              "this run is effectively distributed.", flush=True)


def main():
    if not is_hifloat8_available():
        raise RuntimeError("HiFloat8 unavailable, run on Ascend NPU.")

    args = parse_args()

    # 先初始化 DeepSpeed 通信后端（幂等），之后才能用 dist.get_world_size() 构造 config
    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    is_compile_mode = args.mode in ("compile", "distributed_compile")

    validate_mode(args.mode, world_size)

    # 不同 rank 用不同随机种子，取不同训练样本
    torch.manual_seed(42 + dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
    ).train()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=get_ds_config(
            world_size,
            args.micro_batch,
            args.gradient_accumulation_steps,
            args.lr,
        ),
    )

    if is_compile_mode:
        model_engine = maybe_compile_module(model_engine)

    input_ids, labels = build_dataset(tokenizer, args.seq_len)

    if dist.get_rank() == 0:
        print(f"[MODE] {args.mode} | world_size={world_size} | compile={is_compile_mode} | "
              f"model={args.model_path} | seq_len={args.seq_len}", flush=True)

    run_training(model_engine, input_ids, labels, args.steps)

    if dist.get_rank() == 0:
        print(f"HiFloat8 + DeepSpeed [{args.mode}] passed.", flush=True)


if __name__ == "__main__":
    main()
