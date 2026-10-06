# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MXFP8 weights and activations through FlashInfer's TRTLLM-Gen kernels."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.utils import ActivationType

from ..activation import MoEActivationSupport
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
)
from ..impl_environment import MoEDep
from ..impl_identity import register_moe_impl
from ..interface import _reject
from ..quantization import MXFP8TRTLLMGenFusedMoEMethod
from .base import TrtllmGenFusedMoEBase
from .eligibility import check_no_activation_constants, check_no_expert_bias, check_trtllm_gen_leaf
from .identity import PROVIDER_FLASHINFER, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs


@register_moe_impl
class FlashinferTrtllmGenMxfp8Impl(TrtllmGenFusedMoEBase):
    """FlashInfer-exclusive MXFP8, reached without a provider opt-in flag.

    Logical dimensions stay on the module; physical weight dimensions include
    the TRTLLM-Gen alignment padding. Quantization happens after dispatch so
    communication continues to carry the logical BF16 hidden dimension.
    """

    descriptor = replace(
        trtllm_gen_descriptor(
            PROVIDER_FLASHINFER, "mxfp8", "FlashInfer TRTLLM-Gen MXFP8 with padded MajorK weights."
        ),
        capabilities=MoEStaticCapability(),
    )
    capabilities = descriptor.capabilities
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2})
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(
            cls,
            p,
            d,
            cls._check_mxfp8_path(p, d),
            check_no_expert_bias(cls, p),
            check_no_activation_constants(cls, p),
        )

    @classmethod
    def _check_mxfp8_path(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
        if p.activation_type not in cls.activation_support.kinds:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} supports SwiGLU and ReLU2 only",
            )
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED, "MXFP8 TRTLLM-Gen does not support EPLB"
            )
        if not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                "MXFP8 TRTLLM-Gen requires finalized output",
            )
        if not d.env.has_dep(MoEDep.FLASHINFER_MXFP8_MOE):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires FlashInfer MXFP8 block-scale MoE support",
            )
        if p.hidden_size is not None and p.hidden_size % 32:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED, "MXFP8 hidden size must be divisible by 32"
            )
        if p.intermediate_size is not None and p.intermediate_size % (32 * d.tp_size):
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                "Each MXFP8 TP shard must contain complete 32-element scale blocks",
            )
        return None

    def _requires_separated_routing(self) -> bool:
        return True

    def _check_configs(self) -> None:
        assert self.activation_type in self.activation_support.kinds
        assert (
            not self.bias
            and self.act_alpha is None
            and self.act_beta is None
            and self.act_clamp is None
        )

    def _get_quant_method(self) -> object:
        return MXFP8TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, None]:
        return x, None

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace
        k = prepare_kernel_inputs(self, ctx)
        assert k.do_finalize, "MXFP8 TRTLLM-Gen requires finalized output"
        if k.x.shape[0] == 0:
            return k.x
        hidden = self.w2_weight.shape[1]
        x = F.pad(k.x, (0, hidden - self.hidden_size)).contiguous()
        x, sf = self.op_backend.mxfp8_quantize(x, is_sf_swizzled_layout=False)
        result = self.op_backend.run_mxfp8_moe(
            hidden_states=x,
            hidden_states_scale=sf.view(torch.uint8).reshape(x.shape[0], hidden // 32),
            gemm1_weights=self.w3_w1_weight,
            gemm1_weights_scale=self.w3_w1_weight_scale,
            gemm2_weights=self.w2_weight,
            gemm2_weights_scale=self.w2_weight_scale,
            topk_ids=k.token_selected_experts,
            topk_weights=k.token_final_scales,
            num_experts=self.num_slots,
            top_k=k.top_k,
            intermediate_size=self.w2_weight.shape[2],
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            activation=self.activation_type.name,
            tune_max_num_tokens=self.max_num_tokens,
        )
        if isinstance(result, list):
            result = result[0]
        return result[:, : self.hidden_size].contiguous()
