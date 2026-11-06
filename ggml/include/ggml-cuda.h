#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// The allocation selector is thread-local and must only wrap the intended
// allocation call. Sparse KV buffers reserve a stable CUDA virtual address.
GGML_BACKEND_API void ggml_backend_cuda_set_sparse_kv_alloc(uint32_t n_owners, uint64_t budget_bytes);
GGML_BACKEND_API bool ggml_backend_cuda_buffer_is_sparse_kv(ggml_backend_buffer_t buffer);

// Ensure that resident ranges have private physical storage and accessible
// ranges are readable by the graph. Accessible padding outside resident data
// aliases a shared read-only zero page and consumes no private data page. The
// resident ranges supplied for an owner replace that owner's prior residency
// references. On failure, prior owner references are preserved.
GGML_BACKEND_API bool ggml_backend_cuda_sparse_kv_ensure(
        ggml_backend_buffer_t buffer,
        const uint32_t * owners,
        const size_t * offsets,
        const size_t * resident_sizes,
        const size_t * accessible_sizes,
        size_t n_ranges);
GGML_BACKEND_API bool ggml_backend_cuda_sparse_kv_release_owner(
        ggml_backend_buffer_t buffer,
        uint32_t owner);
// Configure and allocate the fixed physical page pool backing a sparse KV
// virtual buffer. This must be called before the first residency request.
GGML_BACKEND_API bool ggml_backend_cuda_sparse_kv_set_pool_size(
        ggml_backend_buffer_t buffer,
        uint64_t bytes);
// Like set_pool_size, but the pool grows on demand up to `bytes` and trim
// returns unused physical pages to the device (draft-model KV).
GGML_BACKEND_API bool ggml_backend_cuda_sparse_kv_set_elastic_pool_size(
        ggml_backend_buffer_t buffer,
        uint64_t bytes);
// Drop unused virtual mappings back into the fixed pool. No physical CUDA
// allocation is released by this operation.
GGML_BACKEND_API bool ggml_backend_cuda_sparse_kv_trim(ggml_backend_buffer_t buffer);
// Process-wide residency counters: unmap batches (each one synchronizes the device) and their wait, pages unmapped,
// private pages mapped, mapped pages without owners recycled for new data.
GGML_BACKEND_API void ggml_backend_cuda_sparse_kv_stats(uint64_t * unmap_syncs, uint64_t * sync_us, uint64_t * unmapped,
        uint64_t * mapped, uint64_t * recycled);
GGML_BACKEND_API uint64_t ggml_backend_cuda_sparse_kv_pool_bytes(ggml_backend_buffer_t buffer);
GGML_BACKEND_API uint64_t ggml_backend_cuda_sparse_kv_granularity(ggml_backend_buffer_t buffer);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
