# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
import pytest

# isort: off
import torch
# isort: on
import os
import torch.distributed as dist

from parameterized import parameterized

import tensorrt_llm as tllm

from tensorrt_llm.executor.utils import get_spawn_proxy_process_env
from tensorrt_llm.llmapi.utils import print_colored_debug
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

class TestDeepEpSingleGPU(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0x1234)
        tllm.logger.set_level('error')

    # def init_dist(self, local_rank: int, num_local_ranks: int):
    #     # NOTES: you may rewrite this function with your own cluster settings
    #     ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    #     port = int(os.getenv('MASTER_PORT', '8361'))
    #     num_nodes = int(os.getenv('WORLD_SIZE', 1))
    #     node_rank = int(os.getenv('RANK', 0))
    #     print("ip", ip, "port", port, "num_nodes", num_nodes, "node_rank", node_rank)
    #     assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    #     dist.init_process_group(
    #         backend='nccl',
    #         init_method=f'tcp://{ip}:{port}',
    #         world_size=num_nodes * num_local_ranks,
    #         rank=node_rank * num_local_ranks + local_rank
    #     )
    #     torch.set_default_dtype(torch.bfloat16)
    #     torch.set_default_device('cuda')
    #     torch.cuda.set_device(local_rank)

    #     return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

    # def test_loop(self, local_rank: int, num_local_ranks: int, num_nvl_bytes: int, num_rdma_bytes: int, test_ll_compatibility: bool, ll_num_experts: int):
    #     print("local_rank", local_rank, "num_local_ranks", num_local_ranks, "num_nvl_bytes", num_nvl_bytes, "num_rdma_bytes", num_rdma_bytes, "test_ll_compatibility", test_ll_compatibility, "ll_num_experts", ll_num_experts)
    #     rank, num_ranks, group = self.init_dist(local_rank, num_local_ranks)
    #     buffer = torch.ops.trtllm.deepep_buffer(group, num_nvl_bytes, num_rdma_bytes, low_latency_mode=test_ll_compatibility,
    #                             num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1))
    #     return buffer

    @parameterized.expand([
        (4, 1e9, 0, False, 256),
    ])
    def test_deepep_intranode_dispatch(self, num_ranks, num_nvl_bytes, num_rdma_bytes, low_latency_mode, ll_num_experts):
        
        print("num_ranks", num_ranks, "num_nvl_bytes", num_nvl_bytes, "num_rdma_bytes", num_rdma_bytes, "low_latency_mode", low_latency_mode, "ll_num_experts", ll_num_experts)
        mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
        print(f"mpi_process_pre_spawned: {mpi_process_pre_spawned}")
        assert not mpi_process_pre_spawned

        # num_processes = num_ranks
        # torch.multiprocessing.spawn(self.test_loop, args=(num_processes, num_nvl_bytes, num_rdma_bytes, low_latency_mode, ll_num_experts), nprocs=num_processes)

        # print_colored_debug(f"LLM create MpiPoolSession\n",
        #                     "yellow")
        # mpi_session = MpiPoolSession(
        #     n_workers=num_ranks)

        # buffer = torch.ops.trtllm.deepep_buffer(mpi_session.group, num_ranks, num_nvl_bytes, num_rdma_bytes, low_latency_mode)

if __name__ == "__main__":
    unittest.main()
