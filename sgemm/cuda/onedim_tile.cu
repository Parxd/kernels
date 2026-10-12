#include "../../utils.h"

template <int block_M, int block_N, int block_K, int thread_M>
__global__ void onedim_threadtile(int M, int N, int K, float* A, float* B, float* C) {
    // TODO: add assertions for kernel launch parameters
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float A_tile[block_M * block_K];
    __shared__ float B_tile[block_K * block_N];
    int tiles = CEIL_DIV(K, block_K);
    float C_res[thread_M + 1] = {0.0f};  // store intermed. C column & cached B value in TMEM
    for (int tile = 0; tile < tiles; ++tile) {
        // TODO: simplify this complicated GMEM A indexing logic
        A_tile[threadIdx.x * block_K + threadIdx.y] = A[(block_M * blockIdx.y + threadIdx.x) * K + (block_K * tile + threadIdx.y)];
        B_tile[threadIdx.y * block_N + threadIdx.x] = B[(block_K * tile + threadIdx.y) * N + global_idx];
        __syncthreads();
        for (int k = 0; k < block_K; ++k) {
            C_res[thread_M] = B_tile[k * block_N + threadIdx.x];
            for (int thread_row = 0; thread_row < thread_M; ++thread_row) {
                C_res[thread_row] += A_tile[(thread_M * threadIdx.y + thread_row) * block_K + k] * C_res[thread_M];
            }
        }
        __syncthreads();
    }
    // will omit thread boundary checks for sake of simplicity--will assume problem size is always multiple of 32
    for (int thread_row = 0; thread_row < thread_M; ++thread_row) {
        C[(thread_M * global_idy + thread_row) * N + global_idx] = C_res[thread_row];
    }
}

void inline launch_onedim_threadtile(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    dim3 gridDim(CEIL_DIV(M, block_M), CEIL_DIV(N, block_N));
    dim3 blockDim(block_N, block_M / thread_M);
    onedim_threadtile<block_M, block_N, block_K, thread_M><<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK(cudaGetLastError());
}