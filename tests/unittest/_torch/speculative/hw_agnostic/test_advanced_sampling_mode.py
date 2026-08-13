# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for one-model ``advanced_sampling_mode``.

Covers the config contract (enum + skip properties + the use_rejection_sampling
requirement for the top-p-disabling modes), ``resolve_advanced_sampling_filters``
mode resolution, a CUDA check that NO_TOPK yields the same distribution as FULL
when top_k is disabled, and native greedy handling (greedy rows return argmax).
"""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as su
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative import interface as spec_interface
from tensorrt_llm._torch.speculative.interface import (
    SpecMetadata,
    SpeculativeDecodingMode,
    SpecWorkerBase,
)
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, DecodingBaseConfig, MTPDecodingConfig


class _SamplingWorker(SpecWorkerBase):
    @property
    def max_draft_len(self):
        return 3

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _request(*, seed, generated, slot=0, state=LlmRequestState.GENERATION_IN_PROGRESS):
    prompt_len = 100
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[1.0],
            top_k=None,
            top_p=[0.95],
            random_seed=[seed] if seed is not None else None,
        ),
        state=state,
        py_seq_slot=slot,
        py_orig_prompt_len=prompt_len,
        max_beam_num_tokens=prompt_len + generated,
    )


def test_enum_skip_properties():
    """Enum members + the skip properties (single source of truth for filter skipping)."""
    M = AdvancedSamplingMode
    assert [m.value for m in M] == ["full", "no_topk", "no_topp", "no_topk_no_topp"]
    assert (M.FULL.skips_top_k, M.FULL.skips_top_p) == (False, False)
    assert (M.NO_TOPK.skips_top_k, M.NO_TOPK.skips_top_p) == (True, False)
    assert (M.NO_TOPP.skips_top_k, M.NO_TOPP.skips_top_p) == (False, True)
    assert (M.NO_TOPK_NO_TOPP.skips_top_k, M.NO_TOPK_NO_TOPP.skips_top_p) == (True, True)


def test_advanced_sampling_mode_on_base_config():
    """The field lives on DecodingBaseConfig (not MTP-specific) and defaults to FULL."""
    assert "advanced_sampling_mode" in DecodingBaseConfig.model_fields
    assert MTPDecodingConfig(max_draft_len=1).advanced_sampling_mode == AdvancedSamplingMode.FULL


def test_all_modes_construct_regardless_of_rejection():
    """Every mode constructs with or without rejection sampling (no config gating)."""
    for mode in ("full", "no_topk", "no_topp", "no_topk_no_topp"):
        for rej in (False, True):
            cfg = MTPDecodingConfig(
                max_draft_len=1, advanced_sampling_mode=mode, use_rejection_sampling=rej
            )
            assert cfg.advanced_sampling_mode.value == mode


@pytest.mark.parametrize(
    "mode,expect_top_k_none,expect_top_p_none",
    [
        ("full", False, False),
        ("no_topk", True, False),
        ("no_topp", False, True),
        ("no_topk_no_topp", True, True),
    ],
)
def test_resolve_advanced_sampling_filters(mode, expect_top_k_none, expect_top_p_none):
    """Mode resolution None-ifies disabled filters (so the op skips that kernel)
    and passes kept filters through unchanged."""
    top_k = torch.zeros(2, dtype=torch.int32)
    top_p = torch.ones(2)
    eff_top_k, eff_top_p = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode(mode), top_k, top_p
    )
    assert (eff_top_k is None) is expect_top_k_none
    assert (eff_top_p is None) is expect_top_p_none
    if not expect_top_k_none:
        assert eff_top_k is top_k
    if not expect_top_p_none:
        assert eff_top_p is top_p


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize("top_p_val", [1.0, 0.9])
def test_no_topk_matches_full(top_p_val):
    """With top_k disabled, NO_TOPK skips the top_k mask kernel (a no-op at k=vocab)
    and yields the same sampling distribution as FULL. We compare the resulting
    probability distributions rather than the sampled tokens: the flashinfer top_k mask
    at k=vocab injects ~1e-8 fp noise that leaves the distribution unchanged but can flip
    an individual sampled token across GPU archs, so exact-token equality is not portable.
    A real (non-no-op) filter would move mass by orders of magnitude, far above atol."""
    dev = "cuda"
    torch.manual_seed(0)
    batch, vocab = 64, 32000
    logits = torch.randn(batch, vocab, device=dev, dtype=torch.float32) * 2.0
    temperatures = torch.full((batch,), 0.7, device=dev, dtype=torch.float32)
    top_k = torch.zeros(batch, device=dev, dtype=torch.int32)  # disabled
    top_p = torch.full((batch,), top_p_val, device=dev, dtype=torch.float32)

    ek_full, ep_full = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode.FULL, top_k.clone(), top_p
    )
    ek_nt, ep_nt = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode.NO_TOPK, top_k.clone(), top_p
    )
    probs_full = su.compute_probs_from_logits(logits.clone(), temperatures, ek_full, ep_full)
    probs_no_topk = su.compute_probs_from_logits(logits.clone(), temperatures, ek_nt, ep_nt)
    assert torch.allclose(probs_full, probs_no_topk, atol=1e-5, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize("mode", ["no_topk", "no_topk_no_topp"])
def test_greedy_row_returns_argmax_natively(mode):
    """Greedy rows carry the sentinel temperature, so the sampler returns their
    argmax token even in a mixed batch -- this is why no mixed-batch guard is needed."""
    dev = "cuda"
    torch.manual_seed(0)
    batch, vocab = 8, 4096
    logits = torch.randn(batch, vocab, device=dev, dtype=torch.float32) * 3.0
    disable = GREEDY_TEMPERATURE_THRESHOLD / 10  # sentinel for greedy rows
    temperatures = torch.full((batch,), 0.7, device=dev, dtype=torch.float32)
    temperatures[0] = disable  # greedy rows mixed with sampled rows
    temperatures[1] = disable
    top_k = torch.zeros(batch, device=dev, dtype=torch.int32)
    top_p = torch.ones(batch, device=dev, dtype=torch.float32)
    seed = torch.tensor([7], dtype=torch.int64, device=dev)
    offset = torch.tensor([0], dtype=torch.int64, device=dev)

    eff_top_k, eff_top_p = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode(mode), top_k, top_p
    )
    tokens = su.sample_from_logits_op(
        logits, temperatures, eff_top_k, eff_top_p, seed=seed, offset=offset
    )
    argmax = logits.argmax(dim=-1)
    assert tokens[0].item() == argmax[0].item()
    assert tokens[1].item() == argmax[1].item()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize(
    "spec_dec_mode",
    [SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL, SpeculativeDecodingMode.DFLASH],
    ids=["mtp", "dflash"],
)
def test_aligned_target_rng_uses_request_seed_and_output_position(monkeypatch, spec_dec_mode):
    """The diagnostic path assigns each target row the same stateless Philox
    position that regular per-request TorchSampler sampling would use."""
    monkeypatch.setenv("TRTLLM_MTP_ALIGN_TARGET_RNG", "1")
    meta = SpecMetadata(
        max_num_requests=1,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=spec_dec_mode,
        runtime_draft_len=3,
        advanced_sampling_mode=AdvancedSamplingMode.NO_TOPK,
    )
    meta.populate_sampling_params_for_one_model([_request(seed=1234, generated=7)])

    assert meta.target_seeds is not None
    assert meta.target_offsets is not None
    assert meta.target_seeds[:4].tolist() == [1234] * 4
    assert meta.target_offsets[:4].tolist() == [224, 256, 288, 320]

    torch.manual_seed(5)
    logits = torch.randn(4, 4096, dtype=torch.float32, device="cuda")
    worker = _SamplingWorker()
    actual = worker._sample_tokens_for_batch(logits, meta, 0, 1)

    expected = torch.cat(
        [
            su.sample_from_logits_op(
                logits[row : row + 1],
                meta.temperatures[row : row + 1],
                None,
                meta.top_ps[row : row + 1],
                seed=meta.target_seeds[row : row + 1],
                offset=meta.target_offsets[row : row + 1],
            )
            for row in range(4)
        ]
    )
    torch.testing.assert_close(actual, expected)
    # Target sampling must not advance the separate draft-proposal stream.
    assert worker.seed is None
    assert worker.offset is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA metadata buffers")
@pytest.mark.parametrize(
    "spec_dec_mode",
    [SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL, SpeculativeDecodingMode.DFLASH],
    ids=["mtp", "dflash"],
)
def test_aligned_target_rng_batches_by_speculative_depth(monkeypatch, spec_dec_mode):
    """MTP3 and DFlash3 use four request-batched calls, not one per row."""
    monkeypatch.setenv("TRTLLM_MTP_ALIGN_TARGET_RNG", "1")
    meta = SpecMetadata(
        max_num_requests=2,
        max_draft_len=3,
        max_total_draft_tokens=6,
        spec_dec_mode=spec_dec_mode,
        runtime_draft_len=3,
        advanced_sampling_mode=AdvancedSamplingMode.NO_TOPK,
    )
    meta.populate_sampling_params_for_one_model(
        [
            _request(seed=42, generated=7, slot=0),
            _request(seed=42, generated=7, slot=1),
        ]
    )

    calls = []

    def fake_sample(logits, temperatures, top_k, top_p, *, seed, offset):
        calls.append(
            {
                "row_ids": logits[:, 0].to(dtype=torch.int64).cpu().tolist(),
                "seeds": seed.cpu().tolist(),
                "offsets": offset.cpu().tolist(),
            }
        )
        return logits[:, 0].to(dtype=torch.int64)

    monkeypatch.setattr(spec_interface, "sample_from_logits_op", fake_sample)
    logits = torch.arange(16, dtype=torch.float32, device="cuda").reshape(8, 2)
    worker = _SamplingWorker()
    actual = worker._sample_tokens_for_batch(logits, meta, num_contexts=0, batch_size=2)

    assert calls == [
        {"row_ids": [0, 8], "seeds": [42, 42], "offsets": [224, 224]},
        {"row_ids": [2, 10], "seeds": [42, 42], "offsets": [256, 256]},
        {"row_ids": [4, 12], "seeds": [42, 42], "offsets": [288, 288]},
        {"row_ids": [6, 14], "seeds": [42, 42], "offsets": [320, 320]},
    ]
    assert actual.tolist() == [0, 2, 4, 6, 8, 10, 12, 14]
    # Target sampling remains separate from the draft-proposal RNG stream.
    assert worker.seed is None
    assert worker.offset is None


def test_advanced_mode_accepted_on_all_spec_paths():
    """The MTP-one-model-only gate was removed (the field is on the base config),
    so non-FULL modes construct on any spec path instead of raising at config time."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        speculative_config=MTPDecodingConfig(
            max_draft_len=1, use_mtp_vanilla=True, advanced_sampling_mode="no_topk"
        ),
    )
    assert args.speculative_config.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
