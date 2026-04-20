#include <assert.h>
#include "../../utils.h"

// warning: contains a lot of hard-coding, e.g. only using 128-byte transactions
template <int block_M, int block_N, int block_K, int thread_M, int thread_N>
__global__ void vectorize(int M, int N, int K, float* A, float* B, float* C) {
    constexpr int threadblock_size = block_M * block_N / (thread_M * thread_N);
    assert(blockDim.x * blockDim.y == threadblock_size);
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idy = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float A_tile[block_K * block_M];
    __shared__ float B_tile[block_K * block_N];
    const int tiles = CEIL_DIV(K, block_K);
    float res[thread_M * thread_N] = {0.0};
    float A_register[thread_M] = {0.0};
    float B_register[thread_N] = {0.0};
    
    constexpr int A_load_tiles = (block_M * block_K / 4) / threadblock_size;
    constexpr int B_load_tiles = (block_K * block_N / 4) / threadblock_size;
    
    // how many threads needed to cover columns?
    constexpr int A_load_cols = CEIL_DIV(block_K, 4);  // 2 for default params
    constexpr int B_load_cols = CEIL_DIV(block_N, 4);  // 16 for default params
    // how many rows can entire threadblock cover in one load_tile iteration?
    constexpr int A_load_rows = threadblock_size / A_load_cols;  // 32 for default params
    constexpr int B_load_rows = threadblock_size / B_load_cols;  // 4 for default params
    // mapping coordinate (8, 8) -> (32, 2) for default params
    // this mapping would be significantly easier w/ NVIDIA's CUTLASS/CUTE, but will leave this for later kernels
    const int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int A_load_x = linear_idx % A_load_cols;
    const int A_load_y = linear_idx / A_load_cols;
    const int B_load_x = linear_idx % B_load_cols;
    const int B_load_y = linear_idx / B_load_cols;
    
    for (int tile = 0; tile < tiles; ++tile) {
        for (int load_tile = 0; load_tile < A_load_tiles; ++load_tile) {
            float4 vector_load = reinterpret_cast<float4*>(
                &A[(blockIdx.y * block_M + (load_tile * A_load_rows + A_load_y)) * K + (tile * block_K + (A_load_x * 4))]
            )[0];
            A_tile[(A_load_x * 4) * block_M + (A_load_rows * load_tile + A_load_y)] = vector_load.x;
            A_tile[(A_load_x * 4 + 1) * block_M + (A_load_rows * load_tile + A_load_y)] = vector_load.y;
            A_tile[(A_load_x * 4 + 2) * block_M + (A_load_rows * load_tile + A_load_y)] = vector_load.z;
            A_tile[(A_load_x * 4 + 3) * block_M + (A_load_rows * load_tile + A_load_y)] = vector_load.w;
        }
        for (int load_tile = 0; load_tile < B_load_tiles; ++load_tile) {
            reinterpret_cast<float4*>(&B_tile[(load_tile * B_load_rows + B_load_y) * block_N + (B_load_x * 4)])[0] = reinterpret_cast<float4*>(
                &B[(tile * block_K + (load_tile * B_load_rows + B_load_y)) * N + (blockIdx.x * block_N + (B_load_x * 4))]
            )[0];
        }
        __syncthreads();
        for (int k = 0; k < block_K; ++k) {
            for (int m = 0; m < thread_M; ++m) {
                A_register[m] = A_tile[k * block_M + (threadIdx.y * thread_M + m)];
            }
            for (int n = 0; n < thread_N; ++n) {
                B_register[n] = B_tile[k * block_N + (thread_N * threadIdx.x + n)];
            }
            for (int m = 0; m < thread_M; ++m) {
                for (int n = 0; n < thread_N; ++n) {
                    res[m * thread_M + n] += A_register[m] * B_register[n];
                }
            }
        }
        __syncthreads();
    }
    for (int m = 0; m < thread_M; ++m) {
        for (int n = 0; n < thread_N / 4; ++n) {
            reinterpret_cast<float4*>(&C[(global_idy * thread_M + m) * N + (global_idx * thread_N + (n * 4))])[0] = 
                reinterpret_cast<float4*>(&res[m * thread_N + (n * 4)])[0];
        }
    }
}

void inline launch_vectorize(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    constexpr int thread_N = 8;
    assert(block_M / thread_M == block_N / thread_N);
    assert(block_M / thread_M == block_K);

    dim3 blockDim(block_M / thread_M, block_N / thread_N);
    dim3 gridDim(CEIL_DIV(M, block_M), CEIL_DIV(N, block_N));
    vectorize<block_M, block_N, block_K, thread_M, thread_N>
        <<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK(cudaGetLastError());
}