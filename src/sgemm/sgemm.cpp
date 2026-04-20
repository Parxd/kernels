#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "./cuda/naive.cu"
// #include "./cuda/smem.cu"
// #include "./cuda/onedim_tile.cu"
// #include "./cuda/twodim_tile.cu"
// #include "./cuda/vectorize.cu"
#include "./cuda/128x8_cg.cu"
#include "./cuda/128x16.cu"
#include "./cuda/siboehm.cu"

// #include "./cutlass/sgemm_128x16_pipe.cu"
#include "./cutlass/ampere_sgemm_128x32_3stage.cu"

#include <cublas_v2.h>

void launch_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float*B, float beta, float *C) {
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

namespace sgemm {
    void run_kernel(int kernel, int M, int N, int K, float* A, float* B, float* C, cublasHandle_t handle) {
        // CUDA
        if (kernel == 0) launch_naive(M, N, K, A, B, C, nullptr);
        // else if (kernel == 1) launch_smem(M, N, K, A, B, C, nullptr);
        // else if (kernel == 2) launch_onedim_threadtile(M, N, K, A, B, C, nullptr);
        // else if (kernel == 3) launch_twodim_threadtile(M, N, K, A, B, C, nullptr);
        // else if (kernel == 4) launch_vectorize(M, N, K, A, B, C, nullptr);
        else if (kernel == 5) launch_sgemm_128x8_cg(M, N, K, A, B, C, nullptr);
        else if (kernel == 6) launch_sgemm_128x16(M, N, K, A, B, C, nullptr);
        else if (kernel == 7) launch_siboehm(M, N, K, 1.0, A, B, 0.0, C);
        // // CUTLASS
        // else if (kernel == 8) launch_sgemm_128x16_pipe('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N);
        else if (kernel == 9) launch_ampere_sgemm_128x32_3stage('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N);
        // cuBLAS
        else if (kernel == 15) launch_cublas(handle, M, N, K, 1.0, A, B, 0.0, C);
    }  // TODO: maybe separate each kernel to have their own driver boilerplate--CUTLASS kernels get pretty nuanced
}  // namespace sgemm

int main(int argc, char** argv) {
    int kernel = 0;
    int M = 128;
    int N = 128;
    int K = 128;
    bool time = false;
    int trials = 50;
    bool check = false;
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (argc > 1) kernel = std::atoi(argv[1]);
    if (argc > 2) M = std::atoi(argv[2]);
    if (argc > 3) N = std::atoi(argv[3]);
    if (argc > 4) K = std::atoi(argv[4]);
    if (argc > 5) time = bool(std::atoi(argv[5]));
    if (argc > 6) check = bool(std::atoi(argv[6]));
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << "[kernel: int=0-10] [M: int=128] [N: int=128] [K:int=128] [time: bool=0] [verify: bool=0]\n";
        return 0;
    }

    std::cout << "[SGEMM]: Running kernel=" << kernel
              << " with M=" << M
              << ", N=" << N
              << ", K=" << K
              << " (time=" << time
              << ", verify=" << check
              << ")\n";

    auto hA = thrust::host_vector<float>(M * K);
    auto hB = thrust::host_vector<float>(K * N);
    auto hC = thrust::host_vector<float>(M * N);
    // std::iota(hA.begin(), hA.end(), 0.0f);
    // std::iota(hB.begin(), hB.end(), 0.0f);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < M * K; ++i) {
        hA[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < N * K; ++i) {
        hB[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    // fill_ones(hA.data(), hA.size());
    // fill_ones(hB.data(), hB.size());

    auto dA = thrust::device_vector<float>(M * K);
    auto dB = thrust::device_vector<float>(K * N);
    auto dC = thrust::device_vector<float>(M * N, 0);
    dA = hA; dB = hB;

    if (time) {
        for (int i = 0; i < 5; ++i)
            sgemm::run_kernel(kernel, M, N, K, dA.data().get(), dB.data().get(), dC.data().get(), handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> trial_times;
        trial_times.reserve(trials);

        for (int i = 0; i < trials; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            sgemm::run_kernel(kernel, M, N, K, dA.data().get(), dB.data().get(), dC.data().get(), handle);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            trial_times.push_back(ms);
        }

        std::sort(trial_times.begin(), trial_times.end());
        float best_ms   = trial_times.front();
        float median_ms = trial_times[trials / 2];

        double sum = 0.0;
        for (float t : trial_times) sum += t;
        float avg_ms = (float)(sum / trials);

        double var = 0.0;
        for (float t : trial_times) { double d = t - avg_ms; var += d * d; }
        float stddev = (float)std::sqrt(var / (trials - 1));

        double flops = 2.0 * M * N * K;
        double bytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);

        std::cout << "[SGEMM] " << trials << " trials\n";
        std::cout << "  time  : avg " << avg_ms << " ± " << stddev
                << " ms | median " << median_ms << " ms | best " << best_ms << " ms\n";
        std::cout << "  gflop/s: avg " << flops / (avg_ms    * 1e6)
                << " | peak " << flops / (best_ms * 1e6) << "\n";
        std::cout << "  bw.    : " << bytes / (best_ms * 1e6) << " GB/s (at peak)\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    sgemm::run_kernel(kernel, M, N, K, dA.data().get(), dB.data().get(), dC.data().get(), handle);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    hC = dC;

    if (check) {
        auto hC_ref = thrust::host_vector<float>(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += hA[i * K + k] * hB[k * N + j];
                }
                hC_ref[i * N + j] = sum;
            }
        }

        bool failed = false;
        float delta = 0.001;
        std::cout << "[SGEMM]: Verifying results..." << "\n";
        for (uint i = 0; i < hC.size(); ++i) {
            auto diff = std::abs(hC[i] - hC_ref[i]);
            if (diff > delta) {
                std::cerr << "[SGEMM]: Mismatch at (" << i / N << ", " << i % N << "), diff=" << diff << "\n";
                std::cerr << "[SGEMM]: Expected=" << hC_ref[i] << ", Actual=" << hC[i] << "\n";
                failed = true;
            }
        }
        if (failed) return 1;
        else std::cout << "[SGEMM]: All elements match for delta=" << delta << "\n";
    }

    cublasDestroy(handle);
    return 0;
}