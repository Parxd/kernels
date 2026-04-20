#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../utils.h"

#include "cutlass/smem.cu"
#include "cutlass/smem_swizzle.cu"

namespace transpose {
    void run_kernel(int kernel, int m, int n, float* src, float* dst) {
        if (!kernel) {
            launch_smem_transpose(m, n, src, dst);
        }
        else {
            launch_smem_swizzle_transpose(m, n, src, dst);
        }
    }
} // namespace transpose

int main(int argc, char* argv[]) {
    int kernel = 0;
    int M = 256;
    int N = 256;
    bool time = false;
    int trials = 50;
    bool check = false;

    if (argc > 1) kernel = std::atoi(argv[1]);
    if (argc > 2) M = std::atoi(argv[2]);
    if (argc > 3) N = std::atoi(argv[3]);
    if (argc > 4) time = bool(std::atoi(argv[4]));
    if (argc > 5) check = bool(std::atoi(argv[5]));
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0]
                  << " [kernel: int=0-1] [M: int=256] [N: int=256] [time: bool=0] [verify: bool=0]\n";
        return 0;
    }

    std::cout << "[Transpose]: Running kernel=" << kernel
              << " with M=" << M
              << ", N=" << N
              << " (time=" << time
              << ", verify=" << check
              << ")\n";

    auto hA = thrust::host_vector<float>(M * N);
    auto hB = thrust::host_vector<float>(M * N);
    std::iota(hA.begin(), hA.end(), 0.0f);
    // std::srand(static_cast<unsigned>(std::time(nullptr)));
    // for (int i = 0; i < M * N; ++i) {
        // hA[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    // }
    auto dA = thrust::device_vector<float>(M * N);
    auto dB = thrust::device_vector<float>(M * N);
    dA = hA;

    if (time) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        transpose::run_kernel(kernel, M, N, dA.data().get(), dB.data().get());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < trials; ++i) {
            transpose::run_kernel(kernel, M, N, dA.data().get(), dB.data().get());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        float avg_time = milliseconds / trials;

        std::cout << "[Transpose]: Avg. walltime (" << trials << " trials): " << avg_time << " ms\n";       

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    transpose::run_kernel(kernel, M, N, dA.data().get(), dB.data().get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    hB = dB;

    if (check) {
        bool failed = false;
        float delta = 0.001;
        auto hB_ref = thrust::host_vector<float>(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                hB_ref[j * M + i] = hA[i * N + j];
            }
        }        
        for (int i = 0; i < hB.size(); ++i) {
            auto diff = std::abs(hB[i] - hB_ref[i]);
            if (diff > delta) {
                std::cerr << "[Transpose]: Mismatch at (" << i / N << ", " << i % N << "), diff=" << diff << "\n";
                std::cerr << "[Transpose]: Expected=" << hB_ref[i] << ", Actual=" << hB[i] << "\n";
                failed = true;
            }
        }
        if (failed) return 1;
        else std::cout << "[Transpose]: All elements match for delta=" << delta << "\n";
    }

    return 0;
}