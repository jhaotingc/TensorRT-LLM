# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGen static FP8 attention through TRTLLM-gen FMHA."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as trtllm_backend
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="TRTLLM-gen requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 10,
        reason="TRTLLM-gen static FP8 test requires an SM100-family GPU",
    ),
]


def _static_fp8_config() -> QuantAttentionConfig:
    return QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=0,
        k_block_size=0,
        v_block_size=0,
    )


def test_static_fp8_uses_non_sage_trtllm_gen_and_shared_workspace():
    """Static E4M3 uses direct TRTLLM-gen and shares its model-scoped workspace."""
    device = torch.device("cuda")
    batch_size, seq_len, num_heads, head_dim = 1, 256, 4, 128
    scale_values = {"q": 0.01, "k": 0.0125, "v": 0.02}
    scale_tensors = {
        name: torch.tensor(value, dtype=torch.float32, device=device)
        for name, value in scale_values.items()
    }
    shared_state = {}
    backends = [
        trtllm_backend.TrtllmAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            quant_attention_config=_static_fp8_config(),
            attention_metadata_state=shared_state,
        )
        for _ in range(2)
    ]

    torch.manual_seed(321)
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, scale_tensors["q"])
    k_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(k, scale_tensors["k"])
    v_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(v, scale_tensors["v"])
    out_ref = F.scaled_dot_product_attention(
        q_fp8.float().mul(scale_values["q"]).to(torch.bfloat16).transpose(1, 2),
        k_fp8.float().mul(scale_values["k"]).to(torch.bfloat16).transpose(1, 2),
        v_fp8.float().mul(scale_values["v"]).to(torch.bfloat16).transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)
    kwargs = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "attention_mask": PredefinedAttentionMask.FULL,
        "static_q_scale": scale_tensors["q"],
        "static_k_scale": scale_tensors["k"],
        "static_v_scale": scale_tensors["v"],
        "scale_q": scale_values["q"],
        "scale_k": scale_values["k"],
        "scale_v": scale_values["v"],
    }

    direct_fmha = trtllm_backend.trtllm_ragged_attention_deepseek
    with mock.patch.object(
        trtllm_backend,
        "trtllm_ragged_attention_deepseek",
        wraps=direct_fmha,
    ) as mocked_fmha:
        out = backends[0](q, k, v, **kwargs)
        out_prequantized = backends[1](q_fp8, k_fp8, v_fp8, **kwargs)

    assert torch.isfinite(out).all()
    assert out.dtype == torch.bfloat16
    cosine_similarity = F.cosine_similarity(
        out.view_as(out_ref).reshape(-1).float(), out_ref.reshape(-1).float(), dim=0
    ).item()
    assert cosine_similarity > 0.99
    torch.testing.assert_close(out_prequantized, out, atol=0, rtol=0)
    assert len(shared_state["trtllm_gen_static_e4m3"]["workspace"]) == 1

    assert mocked_fmha.call_count == 2
    for call in mocked_fmha.call_args_list:
        assert call.kwargs["backend"] == "trtllm-gen"
        assert call.kwargs["sage_attn_sfs"] == (None, None, None, None)
        assert call.kwargs["num_elts_per_sage_attn_blk"] == (0, 0, 0, 0)
