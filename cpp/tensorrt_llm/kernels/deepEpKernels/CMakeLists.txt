# Find NVSHMEM
# set(NVSHMEM_DIR ${NVSHMEM_DIR})
# find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_DIR}/lib/cmake/nvshmem)

# add_library(nvshmem ALIAS nvshmem::nvshmem)
# add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
# add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

# link_directories(${NVSHMEM_LIB_DIR})

# # Gather all .cpp and .cu files
# file(GLOB_RECURSE SRC_CPP *.cpp)
# file(GLOB_RECURSE SRC_CU *.cu)
# file(GLOB_RECURSE SRC_CUH *.cuh)

# # Create a single library from the gathered source files
# add_library(deep_ep_src OBJECT ${SRC_CPP} ${SRC_CU} ${SRC_CUH})
# set_target_properties(deep_ep_src PROPERTIES
#     POSITION_INDEPENDENT_CODE ON
#     CXX_STANDARD_REQUIRED ON
#     CUDA_STANDARD_REQUIRED ON
#     CXX_STANDARD 14
#     CUDA_STANDARD 14
#     CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(deep_ep_src PUBLIC nvshmem cudart cudadevrt mlx5)

# target_include_directories(deep_ep_src PUBLIC ${NVSHMEM_INCLUDE_DIR})

# set_property(TARGET deep_ep_src PROPERTY POSITION_INDEPENDENT_CODE ON)
# set_property(TARGET deep_ep_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 14
            CUDA_STANDARD 14
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(${target_name} PUBLIC nvshmem cudart cudadevrt mlx5)
endfunction()

add_deep_ep_library(intranode_cuda intranode.cu)
add_deep_ep_library(runtime_cuda runtime.cu)
add_deep_ep_library(internode_cuda internode.cu)
add_deep_ep_library(internode_ll_cuda internode_ll.cu)
# Later, we should link all libraries in `EP_CUDA_LIBRARIES`
set(EP_CUDA_LIBRARIES intranode_cuda runtime_cuda internode_cuda internode_ll_cuda PARENT_SCOPE)