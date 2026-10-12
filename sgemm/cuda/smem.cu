#include "../../utils.h"


template <int tile_size>
__global__ void smem(int M, int N, int K, float* A, float* B, float* C) {
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float A_tile[tile_size * tile_size];
    __shared__ float B_tile[tile_size * tile_size];
    if (global_idx < M && global_idy < N) {
        float tmp = 0.0;
        const int tiles = CEIL_DIV(K, tile_size);
        for (int tile = 0; tile < tiles; ++tile) {
            A_tile[threadIdx.y * tile_size + threadIdx.x] = A[global_idy * K + (tile * tile_size + threadIdx.x)];
            B_tile[threadIdx.y * tile_size + threadIdx.x] = B[(tile * tile_size + threadIdx.y) * N + global_idx];
            __syncthreads();
            for (int k = 0; k < tile_size; ++k) {
                tmp += A_tile[threadIdx.y * tile_size + k] * B_tile[k * tile_size + threadIdx.x];
            }
            __syncthreads();
        }
        C[global_idy * N + global_idx] = tmp;
    }
}

void inline launch_smem(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int tile_size = 32;
    dim3 gridDim(CEIL_DIV(M, tile_size), CEIL_DIV(N, tile_size));
    dim3 blockDim(tile_size, tile_size);
    smem<tile_size><<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
}
