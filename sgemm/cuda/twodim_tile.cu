#include <assert.h>
#include "../../utils.h"

template <int block_M, int block_N, int block_K, int thread_M, int thread_N>
__global__ void twodim_threadtile(int M, int N, int K, float* A, float* B, float* C) {
    assert(blockDim.x * blockDim.y == block_M * block_N / (thread_M * thread_N));
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idy = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float A_tile[block_M * block_K];
    __shared__ float B_tile[block_K * block_N];
    const int tiles = CEIL_DIV(K, block_K);
    float res[thread_M * thread_N] = {0.0};
    float A_register[thread_M] = {0.0};
    float B_register[thread_N] = {0.0};
    // use compile-time values here instead to enable compiler optimizations
    // const int load_K = (blockDim.x * blockDim.y) / block_K;
    constexpr int load_K = ((block_M * block_N) / (thread_M * thread_N)) / block_K;
    constexpr int A_load_tiles = CEIL_DIV(block_M, load_K);
    constexpr int B_load_tiles = CEIL_DIV(block_N, load_K);
    for (int tile = 0; tile < tiles; ++tile) {
        for (int load_tile = 0; load_tile < A_load_tiles; ++load_tile) {
            A_tile[(load_tile * load_K + threadIdx.y) * block_K + threadIdx.x] = 
                A[(blockIdx.y * block_M + (load_tile * load_K + threadIdx.y)) * K + (tile * block_K + threadIdx.x)];
        }
        for (int load_tile = 0; load_tile < B_load_tiles; ++load_tile) {
            B_tile[threadIdx.y * block_N + (load_tile * load_K + threadIdx.x)] =
                B[(tile * block_K + threadIdx.y) * N + (blockIdx.x * block_N + (load_tile * load_K + threadIdx.x))];
        }
        __syncthreads();
        for (int k = 0; k < block_K; ++k) {
            for (int m = 0; m < thread_M; ++m) {
                A_register[m] = A_tile[(thread_M * threadIdx.y + m) * block_K + k];
            }
            for (int n = 0; n < thread_N; ++n) {
                B_register[n] = B_tile[k * block_N + (thread_N * threadIdx.x + n)];
            }
            for (int m = 0; m < thread_M; ++m) {
                for (int n = 0; n < thread_N; ++n) {
                    res[m * thread_N + n] += A_register[m] * B_register[n];
                }
            }
        }
        __syncthreads();
    }
    for (int m = 0; m < thread_M; ++m) {
        for (int n = 0; n < thread_N; ++n) {
            C[(global_idy * thread_M + m) * N + (global_idx * thread_N + n)] = res[m * thread_N + n];
        }
    }
}

void inline launch_twodim_threadtile(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    constexpr int thread_N = 8;
    assert(block_M / thread_M == block_N / thread_N);
    assert(block_M / thread_M == block_K);

    dim3 blockDim(block_M / thread_M, block_N / thread_N);
    dim3 gridDim(CEIL_DIV(M, block_M), CEIL_DIV(N, block_N));
    twodim_threadtile<block_M, block_N, block_K, thread_M, thread_N><<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK(cudaGetLastError());
}