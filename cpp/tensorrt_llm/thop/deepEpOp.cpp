#include "tensorrt_llm/kernels/deepEpKernels/configs.cuh"
#include "tensorrt_llm/kernels/deepEpKernels/api.cuh"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>

#include "deepEpOp.h"

namespace tk = deep_ep;

namespace torch_ext
{

Buffer::Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode):
        rank(int(rank)), num_ranks(int(num_ranks)),
        num_nvl_bytes(num_nvl_bytes), num_rdma_bytes(num_rdma_bytes),
        low_latency_mode(low_latency_mode),
        comm_stream(at::cuda::getStreamFromPool(true)) {
    // Task fifo memory
    int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
    int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
    int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;

    // Common checks
    TORCH_CHECK(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0), "Invalid nvlink buffer size");
    TORCH_CHECK(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()), "Invalid nvshmem buffer size");
    TORCH_CHECK(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode), "Invalid rank setting");
    TORCH_CHECK(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks, less than NVLink peers");
    if (num_rdma_bytes > 0)
        TORCH_CHECK(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode, "Invalid number of ranks, more than NVLink peers");

    // Get ranks
    TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, int(num_ranks / NUM_MAX_NVL_PEERS)), num_nvl_ranks = std::min(int(num_ranks), NUM_MAX_NVL_PEERS);

    // Get device info
    cudaDeviceProp device_prop = {};
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handle
        TLLM_CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes));
        TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + fifo_bytes);

        // Set task fifo
        TORCH_CHECK(NUM_MAX_FIFO_SLOTS % num_nvl_ranks == 0, "Invalid NUM_MAX_FIFO_SLOTS, not divisible by number of NVLink peers");
        task_fifo_ptrs[nvl_rank] = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        task_fifo_ptrs_gpu = reinterpret_cast<int**>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + fifo_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        TLLM_CUDA_CHECK(cudaMemsetAsync(task_fifo_ptrs[nvl_rank], 0, fifo_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    TLLM_CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    TLLM_CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    TLLM_CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    TLLM_CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    TLLM_CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    TLLM_CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        TLLM_CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        TLLM_CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept {
    // Synchronize
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier
        tk::intranode::barrier(task_fifo_ptrs_gpu, head, nvl_rank, num_nvl_ranks, comm_stream);
        move_fifo_slots();
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++ i) if (i != nvl_rank)
                TLLM_CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
        }

        // Free local buffer and error flag
        TLLM_CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
    }

    // Free NVSHMEM
    if (num_rdma_bytes > 0) {
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        tk::internode::barrier();
        tk::internode::free(rdma_buffer_ptr);
        tk::internode::finalize();
    }

    // Free cuBLAS handle, workspace and MoE counter
    TLLM_CUDA_CHECK(cudaFree(workspace));
    TLLM_CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    TLLM_CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));
}

void Buffer::move_fifo_slots(int num_slots) {
    head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS;
}

bool Buffer::is_available() const {
    return available;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::Buffer>("Buffer")
        .def(torch::init<int64_t, int64_t, int64_t, int64_t, bool>())
        .def("is_available", &torch_ext::Buffer::is_available);
}