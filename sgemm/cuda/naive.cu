#include "../../utils.h"

__global__ void naive(int M, int N, int K, float* A, float* B, float* C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < N && idy < M) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[idy * K + i] * B[i * N + idx];
        }
        C[idy * N + idx] = tmp;
    }
}

void inline launch_naive(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    dim3 gridDim(CEIL_DIV(M, block_dim_x), CEIL_DIV(N, block_dim_y));
    dim3 blockDim(block_dim_x, block_dim_y);
    naive<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
}