# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

QUANT_SUPPORTED_TYPES = ("W8A8_DYNAMIC", "W8A16")


def stochastic_round(
    x: torch.Tensor,
    use_random=False,
    seed: int | None = None,
    mode: str | None = None,
    use_env=True,
) -> torch.Tensor:
    """
    Stochastic rounding with a low-overhead deterministic default.

    Modes:
        value_hash: deterministic value-based LCG hash, no arange allocation.
        index_lcg: deterministic index-based LCG, stricter layout alignment but slower.
        fast_rng: uses backend RNG, fastest but not bit-exact across train/rollout.
    """
    env_random = bool(int(os.getenv("USE_STOCHASTIC", "0"))) if use_env else False
    use_random = bool(use_random) or env_random
    if not use_random:
        return torch.round(x)

    floor = torch.floor(x)
    frac = (x - floor).clamp_(0.0, 1.0)

    if seed is None:
        seed = int(os.getenv("STOCHASTIC_ROUND_SEED", "2024"))

    if x.numel() == 0:
        return floor

    mode = (mode or os.getenv("STOCHASTIC_ROUND_MODE", "value_hash")).lower()
    if mode in ("fast", "fast_rng", "rng"):
        rand = torch.rand(x.shape, device=x.device, dtype=torch.float32)
    else:
        if mode in ("index", "index_lcg"):
            state = torch.arange(x.numel(), device=x.device, dtype=torch.int32).reshape(x.shape)
            state = state + int(seed)
        else:
            x_fp32 = x.detach().float().contiguous()
            state = x_fp32.view(torch.int32) + int(seed)

        state = state * 1664525 + 1013904223
        state = state * 1664525 + 1013904223
        rand = state.to(torch.float32)
        rand = torch.where(rand < 0, rand + 4294967296.0, rand)
        rand = rand * 2.3283064365386963e-10

    return floor + (rand < frac).to(dtype=x.dtype)


def quantize_weight_per_channel(
    w_fp: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    qmin: int = -127,
    qmax: int = 127,
    use_npu: bool = False,
) -> torch.Tensor:
    """
    Quantize Linear weight per output channel.

    Args:
        w_fp:        Float weight tensor, shape [out, in]
        scale:       Per-channel scale, shape [out] or [out, 1]
        zero_point:  Per-channel zero point, shape [out] or [out, 1],
                     usually zeros for symmetric quantization
        qmin/qmax:   Quantization range
        use_npu:     Whether to use NPU quantization operator

    Returns:
        Quantized weight tensor, dtype int8, shape [out, in]
    """
    assert w_fp.dim() == 2, "Only supports Linear weight [out, in]"

    # -------- normalize shapes --------
    # if scale.dim() == 2:
    #     scale = scale.squeeze(1)
    
    # if zero_point is not None and zero_point.dim() == 2:
    #     zero_point = zero_point.squeeze(1)

    if zero_point is None:
        zero_point = torch.zeros_like(scale)

    # -------- sanity checks --------
    # if torch.any(scale == 0):
    #     raise RuntimeError("Zero scale encountered in weight quantization")
    #TODO: eps
    if torch.any(scale == 0):
        import warnings
        warnings.warn("Zero scale found, replacing with eps")
        eps = torch.full_like(scale, 1e-5)
        scale = torch.maximum(scale, eps)
    
    # print("================use maximum====================")
    # eps = torch.full_like(scale, 1e-5)
    # scale = torch.maximum(scale, eps)
    # scale = scale.clamp_min(1e-8)

    # -------- quantization --------
    if use_npu:
        # NPU kernel path (bit-exact with Ascend)
        import torch_npu

        # weight shape: [out, in] → per-channel on dim 0
        w_q = torch_npu.npu_quantize(
            w_fp,
            scale,
            zero_point,
            torch.qint8,
            axis=0,
            div_mode=True,
        )
    else:
        # scale_fp32 = scale.float()
        # offset_fp32 = zero_point.float()
        if scale.dim() == 1:
            scale_fp32 = scale_fp32.unsqueeze(1)
        if zero_point.dim() == 1:
            offset_fp32 = offset_fp32.unsqueeze(1)

        # x_fp32 = w_fp.float()
        # x_fp32 = x_fp32 / scale_fp32 + offset_fp32
        x_fp32 = w_fp / scale + zero_point
        x_int = stochastic_round(x_fp32)

        # x_int = torch.clamp(x_int, qmin, qmax)
        x_int = torch.where(x_int < qmin, qmin, x_int)
        x_int = torch.where(x_int > qmax, qmax, x_int)
        w_q = x_int.to(torch.int8)

    return w_q

def apply_weight_only_quantization(state_dict: dict, scale_tensors: dict = None, quant_range: tuple = None, online = False):
    """
    对 state_dict 中的权重进行量化。

    当 scale_tensors 为 None 时（QAT 模式）：
        从 state_dict 自身获取量化参数 (weight_scale, quant_min, quant_max, weight_offset)。
    
    当 scale_tensors 不为 None 时（非 QAT 模式）：
        1. 先将 scale_tensors 中的值（scale, offset）直接覆盖更新到 state_dict 中。
        2. 依据更新后的 state_dict 进行量化计算。
        3. q_min/q_max 由 quant_range 指定。

    Args:
        state_dict: 模型权重字典。
        scale_tensors: 可选的外部量化参数字典，包含 weight_scale 和 weight_offset。
        quant_range: 可选的量化范围 (q_min, q_max)。
    """
    # 如果存在外部 scale_tensors，直接覆盖更新 state_dict


    keys_to_update = []

    for weight_key in list(state_dict.keys()):
        if not weight_key.endswith(".weight"):
            continue

        weight = state_dict[weight_key]

        scale_key = weight_key.replace(".weight", ".weight_scale")
        offset_key = scale_key.replace(".weight_scale", ".weight_offset")


        if scale_tensors is not None:
            # 离线scale
            q_min, q_max = quant_range
            if scale_key in scale_tensors:
                # 量化层
                from torch.distributed._tensor import DTensor, distribute_tensor
                weight_scale = scale_tensors[scale_key]
                weight_offset = scale_tensors[offset_key]
                placements = weight.placements
                device_mesh = weight.device_mesh
                weight_scale = distribute_tensor(
                        weight_scale, 
                        device_mesh, 
                        placements
                    )
                weight_offset = distribute_tensor(
                        weight_offset, 
                        device_mesh, 
                        placements
                    )
            else:
                # 回退层
                continue
        else:
            # 模型自身scale
            if scale_key in state_dict:
                q_min_key = scale_key.replace(".weight_scale", ".quant_min")
                q_max_key = scale_key.replace(".weight_scale", ".quant_max")

                q_min = state_dict[q_min_key]
                q_max = state_dict[q_max_key]
                weight_offset = state_dict[offset_key]

                if online:
                    x_max = weight.abs().float().amax(dim=1)
                    weight_scale = (x_max / q_max).clamp(min=1e-5)
                else:
                    weight_scale = state_dict[scale_key]
                
            else:
                continue
                 
        # 执行量化
        weight = quantize_weight_per_channel(weight, weight_scale, weight_offset, q_min, q_max)
        
        if weight_scale.ndim == 2:
            weight_scale = weight_scale.flatten()
            weight_offset = weight_offset.flatten()
        elif weight_scale.ndim == 3:
            weight_scale = weight_scale.reshape(weight_scale.shape[0], -1)
            weight_offset = weight_offset.reshape(weight_offset.shape[0], -1)

        # import pdb; pdb.set_trace()

        state_dict[weight_key] = weight
        state_dict[scale_key] = weight_scale
        state_dict[offset_key] = weight_offset

    return state_dict


def compare_weight_(state_dict: dict, weight_tensor: dict = None):
    """
    对 state_dict 中的权重进行量化。

    当 scale_tensors 为 None 时（QAT 模式）：
        从 state_dict 自身获取量化参数 (weight_scale, quant_min, quant_max, weight_offset)。
    
    当 scale_tensors 不为 None 时（非 QAT 模式）：
        1. 先将 scale_tensors 中的值（scale, offset）直接覆盖更新到 state_dict 中。
        2. 依据更新后的 state_dict 进行量化计算。
        3. q_min/q_max 由 quant_range 指定。

    Args:
        state_dict: 模型权重字典。
        scale_tensors: 可选的外部量化参数字典，包含 weight_scale 和 weight_offset。
        quant_range: 可选的量化范围 (q_min, q_max)。
    """

    keys_to_update = []

    for scale_key in list(weight_tensor.keys()):
        if not scale_key.endswith(".weight_scale"):
            continue

        weight_key = scale_key.replace(".weight_scale", ".weight")
        
        # 即使 key 在 iterator 中，也要确保 weight 在 state_dict 中存在
        if weight_key not in state_dict:
            continue
        if scale_key not in state_dict:
            print(f"scale of {scale_key} does not exist in state dict, check wheher disable layer is set properly!")
            continue

        weight = state_dict[weight_key]
        init_weight = weight_tensor[weight_key]

        weight_ = weight.full_tensor()

        weight_scale = state_dict[scale_key]

        init_weight_scale = weight_tensor[scale_key]

        weight_scale_ = weight_scale.full_tensor()

        weight_diff = weight_ - init_weight
        abs_diff = weight_diff.abs()

        weight_max_err = abs_diff.max().item()
        weight_mean_err = abs_diff.mean().item()

        scale_diff = weight_scale_ - init_weight_scale
        abs_diff_scale = scale_diff.abs()

        scale_max_err = abs_diff_scale.max().item()
        scale_mean_err = abs_diff_scale.mean().item()

        print(f"{weight_key} compare: max weight error {weight_max_err}; mean weight error {weight_mean_err}; max scale error {scale_max_err}; mean scale error {scale_mean_err}")

        # rank = dist.get_rank()  # 当前进程编号
        # world_size = dist.get_world_size()  # 总进程数
        # shard_weight = weight.to_local()

        # if world_size > 1:
        #     # 收集所有进程的分片
        #     all_shards = [torch.empty_like(shard_weight) for _ in range(world_size)]
        #     dist.all_gather(all_shards, shard_weight)
        #     # 按分片维度拼接（假设按第0维分片，若按其他维度改dim即可）
        #     full_weight = torch.cat(all_shards, dim=0)
        # else:
        #     full_weight = shard_weight  # 单进程无需聚合        



def remove_scale_and_offset(state_dict: dict):
    """
    Remove QAT-only quantization parameters from state_dict.
    quant_min / quant_max are kept.
    """
    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale"):
            offset_key = key.replace(".weight_scale", ".weight_offset")
            ema_key = key.replace(".weight_scale", ".ema_weight_scale")
            aqn_key = key.replace(".weight_scale", ".aqn_noise")
            aqn_step_key = key.replace(".weight_scale", ".aqn_step")
            aqn_last_step_key = key.replace(".weight_scale", ".aqn_last_sample_step")
            
            del state_dict[key]
            if offset_key in state_dict:
                del state_dict[offset_key]
            if ema_key in state_dict:
                del state_dict[ema_key]
            if aqn_key in state_dict:
                del state_dict[aqn_key]
            if aqn_step_key in state_dict:
                del state_dict[aqn_step_key]
            if aqn_last_step_key in state_dict:
                del state_dict[aqn_last_step_key]
    return state_dict


def remove_quant_minmax(state_dict: dict):
    """
    Remove quant_min, quant_max, and QAT-only observer buffers from state_dict.
    weight_scale / weight_offset are kept.
    """
    for key in list(state_dict.keys()):
        if (
            key.endswith(".quant_min")
            or key.endswith(".quant_max")
            or key.endswith(".ema_weight_scale")
            or key.endswith(".aqn_noise")
            or key.endswith(".aqn_step")
            or key.endswith(".aqn_last_sample_step")
        ):
            del state_dict[key]

    return state_dict


def compare_quant_weights(
    runtime_state: dict,
    ckpt_state: dict,
    atol=0.0,
    rtol=0.0,
    max_print=10,
):
    """
    对 runtime 量化权重 和 checkpoint 中的量化权重做逐项对比
    - 对 scale / offset：允许 shape 不同，使用 flatten() 后比较
    - 对其他权重：要求 shape 完全一致
    """
    print("\n====== Quant Weight Comparison ======")

    common_keys = sorted(set(runtime_state.keys()) & set(ckpt_state.keys()))

    if not common_keys:
        print("No common keys found between runtime and checkpoint states.")
        return

    for name in common_keys:
        rt = runtime_state[name]
        ckpt = ckpt_state[name]

        if not torch.is_tensor(rt) or not torch.is_tensor(ckpt):
            continue

        is_scale_or_offset = ("scale" in name) or ("offset" in name)

        # shape 处理
        if is_scale_or_offset:
            rt_cmp = rt.flatten()
            ckpt_cmp = ckpt.flatten()
            if rt_cmp.numel() != ckpt_cmp.numel():
                print(
                    f"[SKIP] {name}: scale/offset numel mismatch "
                    f"{rt_cmp.numel()} vs {ckpt_cmp.numel()}"
                )
                continue
        else:
            if rt.shape != ckpt.shape:
                print(f"[SKIP] {name}: shape mismatch {rt.shape} vs {ckpt.shape}")
                continue
            rt_cmp = rt
            ckpt_cmp = ckpt

        # 对齐 dtype / device
        ckpt_cmp = ckpt_cmp.to(device=rt_cmp.device, dtype=rt_cmp.dtype)

        diff = rt_cmp - ckpt_cmp
        abs_diff = diff.abs()

        max_err = abs_diff.max().item()
        mean_err = abs_diff.mean().item()
        sum_err = abs_diff.sum().item()

        same = torch.allclose(rt_cmp, ckpt_cmp, atol=atol, rtol=rtol)

        status = "OK" if same else "DIFF"
        shape_info = (
            f"rt={tuple(rt.shape)} ckpt={tuple(ckpt.shape)}"
            if is_scale_or_offset else f"{tuple(rt.shape)}"
        )

        print(
            f"[{status}] {name:<60} "
            f"max_err={max_err:.6e} "
            f"mean_err={mean_err:.6e} "
            f"sum_err={sum_err:.6e} "
            f"dtype={rt_cmp.dtype} shape={shape_info}"
        )

        if not same:
            flat = diff.flatten()
            print(f"    first {max_print} diffs: {flat[:max_print].tolist()}")

    print("====== End Comparison ======\n")


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, learn_scale: bool):
        orig_dtype = x.dtype
        if scale is not None:
            x_fp = x / scale + zero_point
            x_int = stochastic_round(x_fp)
            x_int = torch.clamp(x_int, quant_min, quant_max)
            x_hat = (x_int - zero_point) * scale
            x_hat = x_hat.to(orig_dtype)

            ctx.save_for_backward(x, scale, zero_point, x_fp, x_int)
            ctx.other = (quant_min, quant_max, learn_scale)
        else:
            x_max = x.abs().amax(dim=-1, keepdim=True)
            scale = (x_max / quant_max).clamp(min=1e-5)

            use_activation_sr = bool(int(os.getenv("USE_STOCHASTIC_ACTIVATION", "0")))
            x_round = stochastic_round(x / scale, use_random=use_activation_sr, use_env=False)
            x_int = x_round.clamp(quant_min, quant_max)
            x_hat = (x_int * scale).to(orig_dtype)
            ctx.save_for_backward(None, None, None, None, None)
        # import pdb; pdb.set_trace()
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point, x_fp, x_int = ctx.saved_tensors
        if x is not None:
            qmin, qmax, learn_scale = ctx.other

            # ---------- gradient w.r.t x (STE) ----------
            mask = (x_fp >= qmin) & (x_fp <= qmax)
            grad_x = grad_output * mask.to(grad_output.dtype)

            # ---------- gradient w.r.t scale ----------
            if learn_scale:
                grad_scale = torch.zeros_like(scale)

                inside = (x_fp >= qmin) & (x_fp <= qmax)
                below = x_fp < qmin
                above = x_fp > qmax

                grad_scale_inside = (
                    (x_int - zero_point) - (x / scale)
                )

                grad_scale = torch.sum(
                    grad_output * (
                        grad_scale_inside * inside +
                        (qmin - zero_point) * below +
                        (qmax - zero_point) * above
                    ),
                    dim=1,
                    keepdim=True
                )

                # LSQ-style gradient scaling keeps the learnable step size from
                # receiving gradients that grow with the channel width.
                numel_per_scale = max(x.numel() // max(scale.numel(), 1), 1)
                if torch.is_tensor(qmax):
                    qmax_for_grad = qmax.to(device=grad_scale.device, dtype=grad_scale.dtype)
                else:
                    qmax_for_grad = torch.tensor(qmax, device=grad_scale.device, dtype=grad_scale.dtype)
                qmax_for_grad = qmax_for_grad.abs().max().clamp_min(1.0)
                grad_scale = grad_scale * torch.rsqrt(qmax_for_grad * float(numel_per_scale))
            else:
                grad_scale = None

            return grad_x, grad_scale, None, None, None, None
        else:
            return grad_output, None, None, None, None, None

class QuantizedLinearQAT(nn.Module):
    def __init__(
        self,
        linear_module: nn.Linear,
        init_scale=None,
        init_offset=None,
        w_bit=8,
        learn_scale=True,
        name=None,
        is_layer_0=False,
        aqn_enabled=False,
        aqn_sigma_start=0.0,
        aqn_sigma_end=0.0,
        aqn_decay=0.999,
        aqn_resample_steps=1,
    ):
        super().__init__()
        self.weight = nn.Parameter(linear_module.weight.detach().clone())
        self.bias = nn.Parameter(linear_module.bias.detach().clone()) if linear_module.bias is not None else None
        self.name = name
        self.is_layer_0 = is_layer_0
        self.smooth_alpha = 1.0
        self.smooth_reg = 1e-4
        self.ema_scale_decay = 0.99
        self.ema_scale_mix = 0.1
        self.aqn_enabled = bool(aqn_enabled)
        self.aqn_sigma_start = float(aqn_sigma_start)
        self.aqn_sigma_end = float(aqn_sigma_end)
        self.aqn_decay = float(aqn_decay)
        self.aqn_resample_steps = max(int(aqn_resample_steps), 1)
        
        quant_max = 2**(w_bit - 1) - 1
        quant_min = - quant_max
        self.learn_scale = learn_scale
        if init_scale is not None:
            scale = init_scale.to(self.weight.device)
            if scale.dim() == 1:
                scale = scale.unsqueeze(1)
            # FSDP use `use_orig_params=False` which requires that all wrapped modules have uniform `requires_grad`,
            # therefore a buffer instead of parameter is used here to make it forzen.
            # self.weight_scale = nn.Parameter(scale, requires_grad=False)
            self.weight_scale = nn.Parameter(scale)
            self.online = False
        else:
            # per-out-channel scale: [out,1]
            # max_abs = torch.max(self.weight.abs(), dim=-1, keepdim=True).values
            max_abs = self.weight.abs().amax(dim=-1, keepdim=True)
            scale = max_abs / int(quant_max)
            if self.learn_scale:
                self.weight_scale = nn.Parameter(scale)
                self.online = False
            else:
                self.weight_scale = nn.Parameter(scale)
                self.online = True

        # smooth_scale 作用于输入通道(in_features维度)，所以 shape 必须为 [1, in_features]
        # 初始化为 1.0，表示初始状态不进行平滑。网络会在 QAT 训练中自动学习最优平滑因子
        # self.smooth_scale = nn.Parameter(
        #     torch.ones(linear_module.in_features, device=self.weight.device, dtype=self.weight.dtype)
        # )

        self.log_smooth_scale = nn.Parameter(
                    torch.zeros(linear_module.in_features, device=self.weight.device, dtype=self.weight.dtype)
                )
        

        if init_offset is not None:
            weight_offset = init_offset.to(self.weight.device)
            if weight_offset.dim() == 1:
                weight_offset = weight_offset.unsqueeze(1)
        else:
            weight_offset = torch.zeros_like(scale, device=self.weight.device, dtype=scale.dtype)

        self.weight_offset = nn.Parameter(weight_offset)
        self.register_buffer("quant_min", torch.tensor(quant_min, device=scale.device))
        self.register_buffer("quant_max", torch.tensor(quant_max, device=scale.device))
        self.register_buffer("ema_weight_scale", scale.detach().float().clone().clamp_min(1e-5))
        self.register_buffer("aqn_step", torch.zeros((), dtype=torch.long, device=self.weight.device))
        self.register_buffer("aqn_last_sample_step", torch.full((), -1, dtype=torch.long, device=self.weight.device))
        self.register_buffer(
            "aqn_noise",
            torch.zeros(linear_module.in_features, device=self.weight.device, dtype=self.weight.dtype),
        )
        # self.register_buffer(
        #     "smooth_scale",
        #     torch.ones(linear_module.in_features, device=self.weight.device, dtype=self.weight.dtype)
        # )

    # =========================
    # Smooth scale（property：始终最新）
    # =========================
    @property
    def smooth_scale(self):
        return torch.exp(self.log_smooth_scale * self.smooth_alpha).clamp(1e-4, 1e4)

    @torch.no_grad()
    def _update_ema_weight_scale(self, w_smooth):
        if not self.training or not self.learn_scale:
            return

        quant_max = self.quant_max.to(device=w_smooth.device, dtype=torch.float32).abs().clamp_min(1.0)
        cur_scale = w_smooth.detach().abs().float().amax(dim=-1, keepdim=True) / quant_max
        cur_scale = cur_scale.clamp_min(1e-5)
        self.ema_weight_scale.mul_(self.ema_scale_decay).add_(
            cur_scale.to(device=self.ema_weight_scale.device, dtype=self.ema_weight_scale.dtype),
            alpha=1.0 - self.ema_scale_decay,
        )

    def _effective_weight_scale(self):
        scale = self.weight_scale.clamp_min(1e-5)
        if self.learn_scale and self.ema_scale_mix > 0:
            ema_scale = self.ema_weight_scale.to(device=scale.device, dtype=scale.dtype).clamp_min(1e-5)
            scale = scale + (ema_scale - scale).detach() * self.ema_scale_mix
        return scale

    def set_aqn_step(self, step):
        self.aqn_step.fill_(int(step))

    def _current_aqn_sigma(self):
        if not self.aqn_enabled:
            return 0.0
        step = int(self.aqn_step.item())
        return self.aqn_sigma_end + (self.aqn_sigma_start - self.aqn_sigma_end) * (self.aqn_decay ** step)

    @torch.no_grad()
    def _maybe_resample_aqn_noise(self, sigma):
        if sigma <= 0:
            return
        step = int(self.aqn_step.item())
        if int(self.aqn_last_sample_step.item()) == step:
            return
        if step % self.aqn_resample_steps != 0:
            return
        noise = torch.randn(self.aqn_noise.shape, device=self.aqn_noise.device, dtype=torch.float32)
        self.aqn_noise.copy_((noise * float(sigma)).to(dtype=self.aqn_noise.dtype))
        self.aqn_last_sample_step.fill_(step)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x_smooth = x / torch.clamp(self.smooth_scale, min=1e-5)
        # w_smooth = self.weight * torch.clamp(self.smooth_scale, min=1e-5)
        s = self.smooth_scale
        x_smooth = x / s
        if self.training and self.aqn_enabled:
            sigma = self._current_aqn_sigma()
            self._maybe_resample_aqn_noise(sigma)
            if sigma > 0:
                x_smooth = x_smooth + self.aqn_noise.to(device=x_smooth.device, dtype=x_smooth.dtype)
        w_smooth = self.weight * s.unsqueeze(0)
        if self.online:
            scale = None
        else:
            self._update_ema_weight_scale(w_smooth)
            scale = self._effective_weight_scale()

        w_q = FakeQuantize.apply(w_smooth, scale, self.weight_offset, self.quant_min, self.quant_max, self.learn_scale)
        x_ = FakeQuantize.apply(x_smooth, None, self.weight_offset, self.quant_min, self.quant_max, self.learn_scale)
        # import pdb; pdb.set_trace()

        return F.linear(x_, w_q, self.bias)

def extract_layer_index(layer_name):
    """从layer name中提取全局索引"""
    # 匹配形如 model.layers.0.mlp.experts.83.down_proj.weight 的模式
    import re
    match = re.search(r'layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return None

def classify_moe_layer(key):
    """根据key名称判断MoE模型层类型"""
    key_lower = key.lower()
    
    # attention相关层
    attention_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if "q_proj" in key_lower:
        return "q_proj"
    elif "k_proj" in key_lower:
        return "k_proj"
    elif "v_proj" in key_lower:
        return "v_proj"
    elif "o_proj" in key_lower:
        return "o_proj"
    elif "gate_proj" in key_lower:
        return "gate_proj"
    elif "up_proj" in key_lower:
        return "up_proj"
    elif "down_proj" in key_lower:
        return "down_proj"
    else:
        return "other"

def set_qat_aqn_step(module, step):
    for child in module.modules():
        if isinstance(child, QuantizedLinearQAT):
            child.set_aqn_step(step)


def patch_linear(module, scale_tensors=None, prefix="", w_bit=8, learn_scale=True, exclude_linear=None, aqn_config=None):
    exclude_linear = exclude_linear or []
    aqn_config = aqn_config or {}
    for name, child_module in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child_module, nn.Linear) and all(full_name not in excluded for excluded in exclude_linear):
            init_scale = None
            init_offset = None
            layer_idx = extract_layer_index(full_name)
            is_layer_0 = (layer_idx == 0)
            layer_name = classify_moe_layer(full_name)
            if scale_tensors is not None:
                scale_key = f"{full_name}.weight_scale"
                offset_key = f"{full_name}.weight_offset"
                init_scale = scale_tensors.get(scale_key)
                init_offset = scale_tensors.get(offset_key)
            setattr(module, name, QuantizedLinearQAT(
                linear_module=child_module,
                init_scale=init_scale,
                init_offset=init_offset,
                w_bit=w_bit,
                learn_scale=learn_scale,
                name=layer_name,
                is_layer_0=is_layer_0,
                aqn_enabled=aqn_config.get("enabled", False),
                aqn_sigma_start=aqn_config.get("sigma_start", 0.0),
                aqn_sigma_end=aqn_config.get("sigma_end", 0.0),
                aqn_decay=aqn_config.get("decay", 0.999),
                aqn_resample_steps=aqn_config.get("resample_steps", 1),
            ))
            print(f"patching module {full_name} into {getattr(module,name)}")
        if len(list(child_module.named_children())) > 0:
            patch_linear(
                child_module,
                scale_tensors=scale_tensors,
                prefix=full_name,
                w_bit=w_bit,
                learn_scale=learn_scale,
                exclude_linear=exclude_linear,
                aqn_config=aqn_config,
            )


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0
        self.quant_state_dict = None
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for qwen2.5-vl: when using flash_attention_3, set vision tower to use flash_attention_2
        # because the vision tower does not support flash_attention_3
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
            if self.config.actor.scale_source == "calibrated" or self.config.actor.scale_source == "online":
                learn_scale = False
            else:
                learn_scale = True
            # TODO QRL            
            if role == "actor" and self.config.actor.qat:
                aqn_config = {
                    "enabled": self.config.actor.get("aqn_enabled", False),
                    "sigma_start": self.config.actor.get("aqn_sigma_start", 0.0),
                    "sigma_end": self.config.actor.get("aqn_sigma_end", 0.0),
                    "decay": self.config.actor.get("aqn_decay", 0.999),
                    "resample_steps": self.config.actor.get("aqn_resample_steps", 1),
                }
                patch_linear(
                    actor_module,
                    scale_tensors=self.scale_tensors,
                    w_bit=self.config.actor.qat_w_bit,
                    learn_scale=learn_scale,
                    exclude_linear=self.disable_linear,
                    aqn_config=aqn_config,
                )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        # if self._is_rollout and self.config.rollout.name == "hf":
        #     # TODO(zhangchi.usc1992, shengguangming) fix me.
        #     Current, auto_wrap_policy causes HFRollout to hang in Gemma
        #     auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def extract_modules_without_scale_offset(self, model_path):
        """
        从模型JSON中提取仅含weight/bias、不含weight_scale/weight_offset的模块名（如o_proj、down_proj）
        并剔除指定的无关模块（embed_tokens、input_layernorm、norm、post_attention_layernorm）
        
        Args:
            json_input: 可以是JSON文件路径（str），也可以是已加载的JSON字典（dict）
        
        Returns:
            list: 去重、过滤后的目标模块名列表（如['o_proj', 'down_proj']）
        """
        json_input = os.path.join(model_path, "quant_model_description.json")

        target_suffixes = (".weight", ".weight_scale", ".weight_offset")
        # 初始化存储结果的列表
        prefix_list = []
        
        # 读取并加载JSON文件
        try:
            with open(json_input, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：未找到文件 {json_input}")
            return []
        except json.JSONDecodeError:
            print(f"错误：{json_input} 不是合法的JSON文件")
            return []
        
        # 遍历所有键值对，筛选值为FLOAT的键
        for key, value in data.items():
            if value == "FLOAT":
                # 检查键是否以目标后缀结尾
                for suffix in target_suffixes:
                    if key.endswith(suffix):
                        # 提取后缀前的字符串（保留末尾的点）
                        prefix = key[:-len(suffix)]
                        if prefix and prefix not in prefix_list:
                            prefix_list.append(prefix)
                        break  # 匹配到一个后缀即可，无需继续检查
        
        return prefix_list

    def _load_quantization_params(self, model_path):
        """从量化 ckpt 目录加载固定的量化参数。

        加载结果缓存在 self.scale_tensors 和 self.quant_range 中。
        要求 ckpt 目录符合规范，包含 quant_model_description.json文件，其字段 model_quant_type
        须为 QUANT_SUPPORTED_TYPES 之一。ckpt 中存储 weight_scale 和 weight_offset。
        """
        if self.scale_tensors is not None:
            return

        import glob

        from safetensors.torch import load_file

        desc_file = os.path.join(model_path, "quant_model_description.json")
        assert os.path.exists(desc_file), (
            f"Quantization description file not found: {desc_file}. "
            f"Expected quant_model_description.json in the checkpoint directory."
        )
        with open(desc_file, "r") as f:
            quant_desc = json.load(f)
        quant_type = quant_desc.get("model_quant_type")
        if quant_type not in QUANT_SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model_quant_type: {quant_type}. Expected one of {QUANT_SUPPORTED_TYPES}."
            )


        weight_bits = int(quant_type[1])  # "W8..." → 8
        self.quant_range = (
            torch.tensor(-(1 << (weight_bits - 1)), device="npu"),
            torch.tensor((1 << (weight_bits - 1)) - 1, device="npu"),
        )

        quant_keywords = ("weight_scale", "weight_offset")

        self.scale_tensors = {}

        for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            states = load_file(f, device="npu")

            # 保存完整 state，用于精度对齐
            # self.quant_state_dict.update(states)

            self.scale_tensors.update({
                k: v for k, v in states.items()
                if any(kw in k for kw in quant_keywords)
            })
    
    def _load_offline(self, model_path):
        """从量化 ckpt 目录加载固定的量化参数。

        加载结果缓存在 self.scale_tensors 和 self.quant_range 中。
        要求 ckpt 目录符合规范，包含 quant_model_description.json文件，其字段 model_quant_type
        须为 QUANT_SUPPORTED_TYPES 之一。ckpt 中存储 weight_scale 和 weight_offset。
        """
        if self.offline_tensors is not None:
            return

        import glob

        from safetensors.torch import load_file

        desc_file = os.path.join(model_path, "quant_model_description.json")
        assert os.path.exists(desc_file), (
            f"Quantization description file not found: {desc_file}. "
            f"Expected quant_model_description.json in the checkpoint directory."
        )
        with open(desc_file, "r") as f:
            quant_desc = json.load(f)
        quant_type = quant_desc.get("model_quant_type")
        if quant_type not in QUANT_SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model_quant_type: {quant_type}. Expected one of {QUANT_SUPPORTED_TYPES}."
            )

        self.offline_tensors = {}

        for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            states = load_file(f, device="cpu")

            # 保存完整 state，用于精度对齐
            # self.quant_state_dict.update(states)

            self.offline_tensors.update({
                k: v for k, v in states.items()
            })


    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )

        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        import time
        start = time.perf_counter()
        if self.config.rollout.quantization == "ascend":
            # scale_source only matters when doing quantized rollout
            if self.config.actor.scale_source == "calibrated":
                # Calibrated / checkpoint scales (QAT or non-QAT)
                params = apply_weight_only_quantization(
                    params,
                    scale_tensors=self.scale_tensors,
                    quant_range=self.quant_range,
                )
            elif self.config.actor.scale_source == "online":
                # Auto scale (learned during QAT)
                # This branch is only meaningful when actor.qat == True,
                # but behavior is safe even if qat is False.
                params = apply_weight_only_quantization(
                    params,
                    scale_tensors=None,
                    online=True,
                    # quant_range=(-128, 127),
                )
            else:
                params = apply_weight_only_quantization(
                    params,
                    scale_tensors=None,
                    # quant_range=(-128, 127),
                )

        # ---------------------------------------------------------
        # Case 2: Non-quantized rollout
        # ---------------------------------------------------------
        else:
            # QAT model, but rollout does not use quantization
            if self.config.actor.qat:
                params = remove_scale_and_offset(params)

        params = remove_quant_minmax(params)
        end = time.perf_counter()
        dur = (end - start)
        # print(f"============== online quantization time: {dur}s ===============")
        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params


        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        self.scale_tensors = None
        self.quant_range = None
        self.disable_linear = []
        if self.config.rollout.quantization == "ascend":
            self.disable_linear = self.extract_modules_without_scale_offset(self.config.rollout.model_path)
        # Load offline quantization params if needed (calibrated QAT init or non-QAT rollout quantization)
        if (self._is_actor or self._is_rollout) and self.config.actor.scale_source == "calibrated":
            self._load_quantization_params(self.config.rollout.model_path)
            # self._load_offline(self.config.rollout.model_path)
        # TODO: self.quant_range needs to have values even if it is not loaded from ckpt

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            # TiledMLP configuration for memory-efficient MLP computation
            tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            use_tiled_mlp = tiled_mlp_config.get("enabled", False)
            tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)

            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            use_prefix_grouper = hasattr(self.config, "actor") and self.config.actor.get("use_prefix_grouper", False)

            # TiledMLP for ref model: use ref config if specified, otherwise use actor config
            ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
            if ref_tiled_mlp_config is None:
                ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
            ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)

            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=ref_use_tiled_mlp,
                tiled_mlp_shards=ref_tiled_mlp_shards,
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if use_prefix_grouper:
                    self.config.ref.use_prefix_grouper = use_prefix_grouper
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.aqn_step = 0
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            # perform training
            if self.config.actor.get("aqn_enabled", False):
                set_qat_aqn_step(self.actor_module_fsdp, self.aqn_step)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            if self.config.actor.get("aqn_enabled", False):
                self.aqn_step += 1
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time, images_seqlens=images_seqlens
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)


        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        # perform recompute log_prob
        calculate_entropy = not is_lora
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                outputs = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)
            if not is_lora:
                tensors = {"old_log_probs": outputs["log_probs"]}
            else:
                tensors = {"ref_log_prob": outputs["log_probs"]}
            if calculate_entropy:
                tensors["entropys"] = outputs["entropys"]
            if "sum_pi_squared" in outputs:
                tensors["sum_pi_squared"] = outputs["sum_pi_squared"]
            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            outputs = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": outputs["log_probs"]})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)


    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        # TiledMLP configuration for memory-efficient MLP computation
        tiled_mlp_config = config.model.get("tiled_mlp", {})
        use_tiled_mlp = tiled_mlp_config.get("enabled", False)
        tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and config.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `critic.strategy=fsdp2`.")

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.TOKEN_CLS

            else:
                # Convert config to regular Python types before creating PEFT model
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification
                lora_config = {
                    "task_type": TaskType.TOKEN_CLS,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )


        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        override_config = OmegaConf.to_container(OmegaConf.create(config.model.get("override_config", {})))
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=override_config.get("attn_implementation", "flash_attention_2"),
        )
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown", role="compute_rm_score")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
    
    
