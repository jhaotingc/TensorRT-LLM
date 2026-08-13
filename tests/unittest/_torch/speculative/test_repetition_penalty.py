# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.repetition_penalty import (
    OneModelRepetitionPenaltyState,
    apply_linear_spec_repetition_penalty,
    commit_linear_spec_seen_tokens,
    mark_seen_pairs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _seen_tokens(words: torch.Tensor, slot: int, vocab_size: int) -> set[int]:
    words = words[slot].cpu().tolist()
    return {token for token in range(vocab_size) if words[token // 32] & (1 << (token % 32))}


def test_linear_penalty_uses_committed_and_row_local_draft_history():
    vocab_size = 40
    state = OneModelRepetitionPenaltyState.create(
        max_num_requests=2,
        num_seq_slots=4,
        vocab_size=vocab_size,
        device="cuda",
    )
    mark_seen_pairs(
        state.seen_words_cuda,
        torch.tensor([1, 1, 3], device="cuda"),
        torch.tensor([1, 31, 7], device="cuda"),
        vocab_size=vocab_size,
    )
    draft_tokens = torch.tensor([[2, 3], [8, 9]], device="cuda")
    slots = torch.tensor([1, 3], device="cuda")
    repetition = torch.tensor([2.0, 1.5], device="cuda")
    logits = torch.arange(-20, 20, dtype=torch.float32, device="cuda").repeat(6, 1)

    actual = apply_linear_spec_repetition_penalty(
        logits,
        draft_tokens,
        num_contexts=0,
        batch_size=2,
        batch_slot_ids=slots,
        repetition=repetition,
        seen_words=state.seen_words_cuda,
        vocab_size=vocab_size,
    )

    expected = logits.clone()
    histories = [
        {1, 31},
        {1, 2, 31},
        {1, 2, 3, 31},
        {7},
        {7, 8},
        {7, 8, 9},
    ]
    for row, history in enumerate(histories):
        penalty = 2.0 if row < 3 else 1.5
        token_ids = torch.tensor(sorted(history), device="cuda")
        values = expected[row, token_ids]
        expected[row, token_ids] = torch.where(values > 0, values / penalty, values * penalty)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        logits,
        torch.arange(-20, 20, dtype=torch.float32, device="cuda").repeat(6, 1),
    )


def test_factor_one_is_an_exact_noop():
    vocab_size = 64
    state = OneModelRepetitionPenaltyState.create(
        max_num_requests=1,
        num_seq_slots=1,
        vocab_size=vocab_size,
        device="cuda",
    )
    mark_seen_pairs(
        state.seen_words_cuda,
        torch.tensor([0, 0], device="cuda"),
        torch.tensor([1, 63], device="cuda"),
        vocab_size=vocab_size,
    )
    logits = torch.randn(3, vocab_size, dtype=torch.float16, device="cuda")

    actual = apply_linear_spec_repetition_penalty(
        logits,
        torch.tensor([[2, 3]], device="cuda"),
        num_contexts=0,
        batch_size=1,
        batch_slot_ids=torch.tensor([0], device="cuda"),
        repetition=torch.ones(1, device="cuda"),
        seen_words=state.seen_words_cuda,
        vocab_size=vocab_size,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, logits.float(), rtol=0, atol=0)


def test_commit_marks_only_accepted_tokens():
    vocab_size = 64
    state = OneModelRepetitionPenaltyState.create(
        max_num_requests=2,
        num_seq_slots=3,
        vocab_size=vocab_size,
        device="cuda",
    )

    commit_linear_spec_seen_tokens(
        torch.tensor([[4, 5, 6], [31, 32, 63]], device="cuda"),
        torch.tensor([2, 1], dtype=torch.int32, device="cuda"),
        batch_slot_ids=torch.tensor([0, 2], device="cuda"),
        repetition=torch.tensor([1.2, 2.0], device="cuda"),
        seen_words=state.seen_words_cuda,
        dummy_slot_row=state.dummy_slot_row,
        vocab_size=vocab_size,
    )

    assert _seen_tokens(state.seen_words_cuda, 0, vocab_size) == {4, 5}
    assert _seen_tokens(state.seen_words_cuda, 2, vocab_size) == {31}


def test_stage_batch_initializes_prompt_and_clears_reused_slot():
    vocab_size = 64
    state = OneModelRepetitionPenaltyState.create(
        max_num_requests=1,
        num_seq_slots=1,
        vocab_size=vocab_size,
        device="cuda",
    )
    batch_slots = torch.empty(1, dtype=torch.long, device="cuda")
    first_request = SimpleNamespace(
        py_orig_prompt_len=3,
        get_tokens=lambda _: [2, 3, 4, 50],
    )
    state.stage_batch([first_request], [1.2], [0], batch_slots)
    torch.cuda.synchronize()
    assert _seen_tokens(state.seen_words_cuda, 0, vocab_size) == {2, 3, 4}

    second_request = SimpleNamespace(
        py_orig_prompt_len=2,
        get_tokens=lambda _: [8, 9],
    )
    state.stage_batch([second_request], [1.2], [0], batch_slots)
    torch.cuda.synchronize()
    assert _seen_tokens(state.seen_words_cuda, 0, vocab_size) == {8, 9}


def test_populate_allocates_state_before_live_sampling_path():
    vocab_size = 64
    metadata = SpecMetadata(
        max_num_requests=1,
        max_draft_len=2,
        max_total_draft_tokens=2,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        supports_repetition_penalty=True,
        vocab_size=vocab_size,
        num_seq_slots=1,
    )
    metadata.runtime_draft_len = 2
    request = SimpleNamespace(
        sampling_config=SimpleNamespace(
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=[1.2],
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=0,
        py_orig_prompt_len=3,
        get_tokens=lambda _: [4, 5, 6],
    )

    metadata.populate_sampling_params_for_one_model([request])
    torch.cuda.synchronize()

    assert metadata.repetition_state is not None
    assert not metadata.is_all_greedy_sample
    assert metadata.repetition_state.repetition_cuda[0].item() == pytest.approx(1.2)
    assert _seen_tokens(metadata.repetition_state.seen_words_cuda, 0, vocab_size) == {
        4,
        5,
        6,
    }


def test_apply_and_commit_are_cuda_graph_replayable():
    vocab_size = 64
    state = OneModelRepetitionPenaltyState.create(
        max_num_requests=1,
        num_seq_slots=1,
        vocab_size=vocab_size,
        device="cuda",
    )
    logits = torch.randn(3, vocab_size, device="cuda")
    draft_tokens = torch.tensor([[2, 3]], device="cuda")
    slots = torch.tensor([0], device="cuda")
    repetition = torch.tensor([1.1], device="cuda")
    accepted = torch.tensor([[4, 5, 6]], device="cuda")
    num_accepted = torch.tensor([2], dtype=torch.int32, device="cuda")

    # Compile the Triton kernels before capture.
    apply_linear_spec_repetition_penalty(
        logits,
        draft_tokens,
        num_contexts=0,
        batch_size=1,
        batch_slot_ids=slots,
        repetition=repetition,
        seen_words=state.seen_words_cuda,
        vocab_size=vocab_size,
    )
    commit_linear_spec_seen_tokens(
        accepted,
        num_accepted,
        batch_slot_ids=slots,
        repetition=repetition,
        seen_words=state.seen_words_cuda,
        dummy_slot_row=state.dummy_slot_row,
        vocab_size=vocab_size,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply_linear_spec_repetition_penalty(
            logits,
            draft_tokens,
            num_contexts=0,
            batch_size=1,
            batch_slot_ids=slots,
            repetition=repetition,
            seen_words=state.seen_words_cuda,
            vocab_size=vocab_size,
        )
        commit_linear_spec_seen_tokens(
            accepted,
            num_accepted,
            batch_slot_ids=slots,
            repetition=repetition,
            seen_words=state.seen_words_cuda,
            dummy_slot_row=state.dummy_slot_row,
            vocab_size=vocab_size,
        )

    logits.copy_(torch.randn_like(logits))
    graph.replay()
    torch.cuda.synchronize()
    assert output.isfinite().all()
    assert {4, 5}.issubset(_seen_tokens(state.seen_words_cuda, 0, vocab_size))
