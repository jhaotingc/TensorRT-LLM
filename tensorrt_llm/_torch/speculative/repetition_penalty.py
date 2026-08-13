# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-side repetition penalty for linear one-model speculation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from ..pyexecutor.llm_request import LlmRequest


def get_repetition_penalty(sampling_config) -> float:
    """Return a request's scalar repetition penalty, including its default."""
    if sampling_config is None:
        return 1.0
    values = getattr(sampling_config, "repetition_penalty", None)
    return 1.0 if not values else float(values[0])


@triton.jit
def _mark_seen_pairs_kernel(
    seen_words_ptr,
    slots_ptr,
    tokens_ptr,
    seen_words_stride,
    vocab_size,
    num_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    pair_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_pair = pair_idx < num_pairs
    slot = tl.load(slots_ptr + pair_idx, mask=valid_pair, other=0).to(tl.int64)
    token = tl.load(tokens_ptr + pair_idx, mask=valid_pair, other=-1).to(tl.int64)
    valid_token = valid_pair & (token >= 0) & (token < vocab_size)
    word = token // 32
    bit = token % 32
    bit_mask = (1 << bit).to(tl.int32)
    tl.atomic_or(
        seen_words_ptr + slot * seen_words_stride + word,
        bit_mask,
        mask=valid_token,
    )


def mark_seen_pairs(
    seen_words: torch.Tensor,
    slots: torch.Tensor,
    tokens: torch.Tensor,
    *,
    vocab_size: int,
) -> None:
    """Mark flattened ``(slot, token)`` pairs in a persistent seen bitset."""
    assert slots.ndim == 1 and tokens.ndim == 1
    assert slots.shape == tokens.shape
    assert slots.device == seen_words.device and tokens.device == seen_words.device
    if slots.numel() == 0:
        return
    block_size = 256
    _mark_seen_pairs_kernel[(triton.cdiv(slots.numel(), block_size),)](
        seen_words,
        slots,
        tokens,
        seen_words.stride(0),
        vocab_size,
        slots.numel(),
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _linear_spec_repetition_kernel(
    logits_ptr,
    logits_stride,
    draft_tokens_ptr,
    draft_stride,
    batch_slot_ids_ptr,
    repetition_ptr,
    seen_words_ptr,
    seen_words_stride,
    vocab_size,
    num_contexts: tl.constexpr,
    draft_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    is_context = row < num_contexts
    generation_row = tl.maximum(row - num_contexts, 0)
    generation_idx = generation_row // (draft_len + 1)
    local_pos = generation_row % (draft_len + 1)
    request_idx = tl.where(is_context, row, num_contexts + generation_idx)

    slot = tl.load(batch_slot_ids_ptr + request_idx).to(tl.int64)
    penalty = tl.load(repetition_ptr + request_idx)
    vocab_ids = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = vocab_ids < vocab_size

    word = vocab_ids // 32
    bit = vocab_ids % 32
    packed = tl.load(
        seen_words_ptr + slot * seen_words_stride + word,
        mask=valid,
        other=0,
    )
    seen = (packed & (1 << bit).to(tl.int32)) != 0

    # The K+1 verification rows see progressively longer prefixes of the
    # current draft. These are temporary predicates only; rejected suffixes
    # must never enter the persistent bitset.
    for pos in tl.static_range(0, draft_len):
        previous = tl.load(
            draft_tokens_ptr + generation_idx * draft_stride + pos,
            mask=(~is_context) & (pos < local_pos),
            other=-1,
        )
        seen |= (~is_context) & (pos < local_pos) & (vocab_ids == previous)

    value = tl.load(
        logits_ptr + row * logits_stride + vocab_ids,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    penalized = tl.where(value > 0.0, value / penalty, value * penalty)
    apply = seen & (penalty != 1.0)
    tl.store(
        logits_ptr + row * logits_stride + vocab_ids,
        tl.where(apply, penalized, value),
        mask=valid,
    )


def apply_linear_spec_repetition_penalty(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    num_contexts: int,
    batch_size: int,
    batch_slot_ids: torch.Tensor,
    repetition: torch.Tensor,
    seen_words: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Apply prompt/history plus row-local draft-prefix repetition penalty.

    The returned FP32 tensor is distinct from ``logits`` so workers can keep
    exposing the unprocessed target logits in their output contract.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert draft_tokens.ndim == 2
    draft_len = draft_tokens.shape[1]
    num_generations = batch_size - num_contexts
    expected_rows = num_contexts + num_generations * (draft_len + 1)
    assert logits.shape[0] == expected_rows, (
        f"linear speculative logits rows mismatch: {logits.shape[0]} != {expected_rows}"
    )
    assert logits.shape[1] == vocab_size, (
        "linear speculative repetition penalty currently requires full target "
        f"vocabulary logits: {logits.shape[1]} != {vocab_size}"
    )
    assert draft_tokens.shape[0] == num_generations
    assert batch_slot_ids.shape[0] >= batch_size
    assert repetition.shape[0] >= batch_size

    processed_logits = torch.empty_like(logits, dtype=torch.float32)
    processed_logits.copy_(logits)
    block_size = 8192
    grid = (
        processed_logits.shape[0],
        triton.cdiv(processed_logits.shape[1], block_size),
    )
    _linear_spec_repetition_kernel[grid](
        processed_logits,
        processed_logits.stride(0),
        draft_tokens,
        draft_tokens.stride(0),
        batch_slot_ids,
        repetition,
        seen_words,
        seen_words.stride(0),
        vocab_size,
        num_contexts=num_contexts,
        draft_len=draft_len,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return processed_logits


@triton.jit
def _commit_seen_tokens_kernel(
    accepted_tokens_ptr,
    accepted_tokens_stride,
    num_accepted_tokens_ptr,
    batch_slot_ids_ptr,
    repetition_ptr,
    seen_words_ptr,
    seen_words_stride,
    vocab_size,
    dummy_slot_row,
):
    request_idx = tl.program_id(0).to(tl.int64)
    position = tl.program_id(1).to(tl.int64)
    slot = tl.load(batch_slot_ids_ptr + request_idx).to(tl.int64)
    penalty = tl.load(repetition_ptr + request_idx)
    valid = position < tl.load(num_accepted_tokens_ptr + request_idx)
    valid &= (slot != dummy_slot_row) & (penalty != 1.0)
    token = tl.load(
        accepted_tokens_ptr + request_idx * accepted_tokens_stride + position,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= (token >= 0) & (token < vocab_size)
    word = token // 32
    bit = token % 32
    tl.atomic_or(
        seen_words_ptr + slot * seen_words_stride + word,
        (1 << bit).to(tl.int32),
        mask=valid,
    )


def commit_linear_spec_seen_tokens(
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    batch_slot_ids: torch.Tensor,
    repetition: torch.Tensor,
    seen_words: torch.Tensor,
    dummy_slot_row: int,
    vocab_size: int,
) -> None:
    """Commit only tokens emitted by linear speculative acceptance."""
    assert accepted_tokens.ndim == 2
    batch_size, max_accepted = accepted_tokens.shape
    assert num_accepted_tokens.shape[0] >= batch_size
    _commit_seen_tokens_kernel[(batch_size, max_accepted)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        batch_slot_ids,
        repetition,
        seen_words,
        seen_words.stride(0),
        vocab_size,
        dummy_slot_row,
    )


@dataclass
class OneModelRepetitionPenaltyState:
    """Persistent repetition-only history shared by eager and graph metadata."""

    seen_words_cuda: torch.Tensor
    repetition_cuda: torch.Tensor
    slot_owner_requests: list[object | None]
    vocab_size: int
    dummy_slot_row: int

    @classmethod
    def create(
        cls,
        *,
        max_num_requests: int,
        num_seq_slots: int,
        vocab_size: int,
        device: torch.device | str,
    ) -> "OneModelRepetitionPenaltyState":
        slot_capacity = num_seq_slots or max_num_requests
        with torch.inference_mode(False):
            seen_words_cuda = torch.zeros(
                (slot_capacity + 1, triton.cdiv(vocab_size, 32)),
                dtype=torch.int32,
                device=device,
            )
            repetition_cuda = torch.ones(
                max_num_requests,
                dtype=torch.float32,
                device=device,
            )
        return cls(
            seen_words_cuda=seen_words_cuda,
            repetition_cuda=repetition_cuda,
            slot_owner_requests=[None] * slot_capacity,
            vocab_size=vocab_size,
            dummy_slot_row=slot_capacity,
        )

    def stage_batch(
        self,
        requests: Sequence["LlmRequest"],
        repetition_factors: Sequence[float],
        slot_ids: Sequence[int],
        batch_slot_ids: torch.Tensor,
    ) -> None:
        """Stage current factors/slots and initialize newly owned prompt rows."""
        batch_size = len(requests)
        assert len(repetition_factors) == batch_size
        assert len(slot_ids) == batch_size
        assert batch_size <= self.repetition_cuda.numel()
        assert batch_size <= batch_slot_ids.numel()

        staged_repetition: list[float] = []
        clear_slots: list[int] = []
        prompt_slots: list[int] = []
        prompt_tokens: list[int] = []

        for request, repetition, slot in zip(requests, repetition_factors, slot_ids):
            if slot == self.dummy_slot_row:
                staged_repetition.append(1.0)
                continue
            if not 0 <= slot < self.dummy_slot_row:
                raise ValueError(f"py_seq_slot {slot} is outside [0, {self.dummy_slot_row})")

            staged_repetition.append(float(repetition))
            if self.slot_owner_requests[slot] is request:
                continue

            self.slot_owner_requests[slot] = request
            clear_slots.append(slot)
            if repetition == 1.0:
                continue
            prompt = list(request.get_tokens(0))[: request.py_orig_prompt_len]
            for token in prompt:
                token = int(token)
                if 0 <= token < self.vocab_size:
                    prompt_slots.append(slot)
                    prompt_tokens.append(token)

        slots_host = torch.tensor(
            slot_ids,
            dtype=torch.long,
            pin_memory=prefer_pinned(),
        )
        repetition_host = torch.tensor(
            staged_repetition,
            dtype=torch.float32,
            pin_memory=prefer_pinned(),
        )
        batch_slot_ids[:batch_size].copy_(slots_host, non_blocking=True)
        self.repetition_cuda[:batch_size].copy_(repetition_host, non_blocking=True)

        if clear_slots:
            clear_slots_cuda = torch.tensor(
                clear_slots,
                dtype=torch.long,
                pin_memory=prefer_pinned(),
            ).to(self.seen_words_cuda.device, non_blocking=True)
            self.seen_words_cuda.index_fill_(0, clear_slots_cuda, 0)

        if prompt_slots:
            prompt_slots_cuda = torch.tensor(
                prompt_slots,
                dtype=torch.long,
                pin_memory=prefer_pinned(),
            ).to(self.seen_words_cuda.device, non_blocking=True)
            prompt_tokens_cuda = torch.tensor(
                prompt_tokens,
                dtype=torch.long,
                pin_memory=prefer_pinned(),
            ).to(self.seen_words_cuda.device, non_blocking=True)
            mark_seen_pairs(
                self.seen_words_cuda,
                prompt_slots_cuda,
                prompt_tokens_cuda,
                vocab_size=self.vocab_size,
            )
