#include "../../utils.h"

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int TM, int TN, int TK,
          int WIM, int WIN,
          int thrs, int blks>
__global__ void __launch_bounds__(thrs, blks) sgemm_128x16(
    int M, int N, int K, float alpha, float* A, float* B, float beta, float* C
) {
    __shared__ float shared_A[BM * BK];
    __shared__ float shared_B[BK * BN];
    float reg_A[TM * TK * WIM];
    float reg_B[TK * TN * WIN];
    float reg_C[TM * TN * WIM * WIN] = {0.0};

    constexpr int gA_lds = (BM * BK) / thrs / 4;
    constexpr int gA_ld_rows = (thrs * 4) / BK;
    constexpr int gB_lds = (BK * BN) / thrs / 4;
    constexpr int gB_ld_rows = thrs / (BN / 4);

    int thread_row_A = threadIdx.x / (BK / 4);
    int thread_col_A = threadIdx.x % (BK / 4);
    // for 2nd loading schema
    // int thr_row_A = (threadIdx.x / BK) * 4;
    // int thr_col_A = threadIdx.x % BK;

    int thread_row_B = threadIdx.x / (BN / 4);
    int thread_col_B = threadIdx.x % (BN / 4);

    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / (BN / WN);
    int warp_col = warp_id % (BN / WN);
    int thr_row = (threadIdx.x % 32) / (WN / WIN / TN);
    int thr_col = (threadIdx.x % 32) % (WN / WIN / TN);
    constexpr int warptiles = BK / WK;
    constexpr int stride_m = WM / WIM;
    constexpr int stride_n = WN / WIN;

    float* global_A = &A[blockIdx.y * BM * K];
    float* global_B = &B[blockIdx.x * BN];
    int tiles = K / BK;
    for (int tile = 0; tile < tiles; ++tile) {
        /*
        1st loading schema: ld.global.v4.f32 -> st.shared.f32
        benchmarking shows higher relative band4 on lower BK = 16 sizes (highest absolute band4)
        */
#pragma unroll
        for (int ld = 0; ld < gA_lds; ++ld) {
            float* global_thread = &global_A[(ld * gA_ld_rows + thread_row_A) * K + (tile * BK + (thread_col_A * 4))];
            auto load = reinterpret_cast<float4*>(global_thread)[0];
            shared_A[(thread_col_A * 4) * BM + (ld * gA_ld_rows + thread_row_A)] = load.x;
            shared_A[(thread_col_A * 4 + 1) * BM + (ld * gA_ld_rows + thread_row_A)] = load.y;
            shared_A[(thread_col_A * 4 + 2) * BM + (ld * gA_ld_rows + thread_row_A)] = load.z;
            shared_A[(thread_col_A * 4 + 3) * BM + (ld * gA_ld_rows + thread_row_A)] = load.w;
        }
        /*
        2nd loading schema: ld.global.f32 -> st.shared.v4.f32
        benchmarking shows higher relative band4 on higher BK = 32 sizes
        */
// #pragma unroll
//         for (int ld = 0; ld < gA_lds; ++ld) {
//             float4 load;
//             load.x = global_A[(ld * gA_ld_rows + thr_row_A) * K + (tile * BK + thr_col_A)];
//             load.y = global_A[(ld * gA_ld_rows + thr_row_A + 1) * K + (tile * BK + thr_col_A)];
//             load.z = global_A[(ld * gA_ld_rows + thr_row_A + 2) * K + (tile * BK + thr_col_A)];
//             load.w = global_A[(ld * gA_ld_rows + thr_row_A + 3) * K + (tile * BK + thr_col_A)];
//             reinterpret_cast<float4*>(&shared_A[thr_col_A * BM + (ld * gA_ld_rows + thr_row_A)])[0] = load;
//         }

#pragma unroll
        for (int ld = 0; ld < gB_lds; ++ld) {
            float* global_thread = &global_B[(tile * BK + (ld * gB_ld_rows + thread_row_B)) * N + (thread_col_B * 4)];
            auto load = reinterpret_cast<float4*>(global_thread)[0];
            reinterpret_cast<float4*>(&shared_B[(ld * gB_ld_rows + thread_row_B) * BN + (thread_col_B * 4)])[0] = load;
        }
        __syncthreads();

#pragma unroll
        for (int warptile = 0; warptile < warptiles; ++warptile) {
#pragma unroll
            for (int warp_m = 0; warp_m < WIM; ++warp_m) {
                float* gen_addr = &shared_A[warptile * BM + (warp_row * WM + (warp_m * stride_m + (thr_row * TM + 0)))];
                auto shared_addr = __cvta_generic_to_shared(gen_addr);
                // asm volatile(
                //     "cvta.to.shared.u64 %0, %1;" 
                //     : "=l"(shared_addr) 
                //     : "l"(gen_addr)
                // );
                asm volatile (
                    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(reg_A[warp_m * TM]), "=f"(reg_A[warp_m * TM + 1]), "=f"(reg_A[warp_m * TM + 2]), "=f"(reg_A[warp_m * TM + 3])
                    : "l"(shared_addr)
                );
            }
#pragma unroll
            for (int warp_n = 0; warp_n < WIN; ++warp_n) {
                float* gen_addr = &shared_B[warptile * BN + (warp_col * WN + (warp_n * stride_n + (thr_col * TN)))];
                auto shared_addr = __cvta_generic_to_shared(gen_addr);
                asm volatile (
                    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(reg_B[warp_n * TN]), "=f"(reg_B[warp_n * TN + 1]), "=f"(reg_B[warp_n * TN + 2]), "=f"(reg_B[warp_n * TN + 3])
                    : "l"(shared_addr)
                );
                // if (!tile && thread(0)) printf("%f\n", reg_B[0]);
            }
            // MMA
#pragma unroll
            for (int warp_m = 0; warp_m < WIM; ++warp_m) {
#pragma unroll
                for (int warp_n = 0; warp_n < WIN; ++warp_n) {
#pragma unroll
                    for (int thread_m = 0; thread_m < TM; ++thread_m) {
#pragma unroll
                        for (int thread_n = 0; thread_n < TN; ++thread_n) {
                            reg_C[(warp_m * TM + thread_m) * (TN * WIN) + (warp_n * TN + thread_n)] += reg_A[warp_m * TM + thread_m] * reg_B[warp_n * TN + thread_n];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    int thr_st_row = (threadIdx.x % 32) / 4;
    int thr_st_col = (threadIdx.x % 32) % 4;
#pragma unroll
    for (int warp_m = 0; warp_m < WIM; ++warp_m) {
#pragma unroll
        for (int warp_n = 0; warp_n < WIN; ++warp_n) {
#pragma unroll
            for (int thread_m = 0; thread_m < TM; ++thread_m) {
                auto st_idx = (blockIdx.y * BM + (warp_row * WM + (warp_m * (WM / WIM) + (thr_st_row * 4 + thread_m)))) * N + (blockIdx.x * BN + (warp_col * WN + (warp_n * (WN / WIN) + (thr_st_col * 4))));
                auto global = reinterpret_cast<float4*>(&C[st_idx])[0];
                auto local = reinterpret_cast<float4*>(&reg_C[(warp_m * TM + thread_m) * (TN * WIN) + (warp_n * TN)])[0];
                float4 write;
                write.x = alpha * local.x + beta * global.x;
                write.y = alpha * local.y + beta * global.y;
                write.z = alpha * local.z + beta * global.z;
                write.w = alpha * local.w + beta * global.w;
                reinterpret_cast<float4*>(&C[st_idx])[0] = write;
            }
        }
    }
}

__host__ inline void launch_sgemm_128x16(
    int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream
) {
    // for CC 8.6 ONLY
    // see https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
    constexpr int regfile_size_sm = 65536;  // unit: 32-bit registers
    constexpr int smem_bytes_sm = 102400;
    constexpr int threads_sm = 1536;

    // assuming NN row-major GEMM problem
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 1;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int TK = 1;

    // warp-level
    constexpr int warptile_size = WM * WN;
    constexpr int threadtile_size = warptile_size / 32;
    constexpr int threadtiles = threadtile_size / (TM * TN);
    constexpr int WIM = 2;  // num. "rows" when tiling over warptile
    constexpr int WIN = threadtiles / WIM;  // num. "cols" when tiling over warptile

    // block-level
    constexpr int warps = (BM / WM) * (BN / WN);
    constexpr int threads = warps * 32;
    constexpr int registers = threads * ((TM * TK * WIM) + (TK * TN * WIN) + (TM * TN * WIM * WIN));
    constexpr int smem_bytes = sizeof(float) * ((BM * BK) + (BK * BN));
    constexpr int min_blocks = std::min({
        smem_bytes_sm / smem_bytes,
        regfile_size_sm / registers,
        threads_sm / threads
    });

    static_assert(threads <= 1024);
    static_assert(registers / threads <= 255);  // for CC x.x: max 255 32-bit reg. per thread allowed
    static_assert(!(BM % WM));
    static_assert(!(BN % WN));
    static_assert(!(BK % WK));
    static_assert(!(WM % TM));
    static_assert(!(WN % TN));
    static_assert(!(WK % TK));

    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    constexpr dim3 block_dim(threads);
    sgemm_128x16<BM, BN, BK,
                   WM, WN, WK,
                   TM, TN, TK,
                   WIM, WIN,
                   threads, min_blocks>
                   <<<grid_dim, block_dim, 0, stream>>>
                   (M, N, K, 1.0, A, B, 0.0, C);
    
    /*
        (BM, BN, BK) = (128, 128, 8)
        (WM, WN, WK) = (64, 64, 4)
        (TM, TN, TK) = (4, 4, 2)

        sA = 128 * 8 = 1024
        sB = 8 * 128 = 1024
        ~ 2048 fp32 / CTA = 8.192 kB / CTA
        100 kB / SM --> 100 / 8.192 = 12 CTA / SM

        rA = 4 * 2 = 8
        rB = 2 * 4 = 8
        rC = 4 * 4 = 16
        ~ 32 register / thread
        
        (128 / 64) * (128 / 64) = 4 warp / CTA
        4 * 32 = 128 thread / CTA
        128 * 32 = 4096 register / CTA
        65536 / 4096 = 16 CTA / SM

        shared memory is the bottleneck
    */

    /*
        how does WK / TK affect occupancy?
        suppose WK / TK = 4 --> threads perform 4 (unrolled) loop iters. over WK for MMA --> lower register pressure
        suppose WK / TK = 1 --> higher register pressure to store (TM * TK) + (TK * TN) fp32
        we can afford higher register pressure, since SMEM is currently bottleneck

        what about BK / WK?
        suppose BK / WK = 2 --> warps perform 2 iters. over BK
        a higher WK could mean lower register pressure if TK is held constant

        suppose all other params. held constant, but TK = 4
        (4 * 4) + (4 * 4) + (4 * 4) = 48 register / thread

        128 thread / CTA * 48 register / thread = 6144 register / CTA
        65536 / 6144 = 10 CTA / SM

        registerfile is now the bottleneck --> if registerfile bottleneck results in less CTAs per SM, what's the benefit?
        each thread covers WK in one iteration, higher compute intensity / thread + higher ILP on CUDA cores
    */
}

#if 0
void occupancy_test() {
    constexpr int regfile_size_sm = 65536;  // unit: 32-bit registers
    constexpr int smem_bytes_sm = 102400;
    constexpr int threads_sm = 1536;
    constexpr int warps_sm = threads_sm / 32;

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;

    // block-level
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    constexpr int WIM = 2;  // rows when warps tiled for MMA
    constexpr int WIN = warps / WIM;  // cols. when warps tiled for MMA

    constexpr int WM = BM / WIM;
    constexpr int WN = BN / WIN;
    constexpr int mma_units = (WM * WN) / 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int threadtiles = mma_units / (TM * TN);
    constexpr int TIM = 2;
    constexpr int TIN = threadtiles / TIM;

    constexpr int smem_bytes = sizeof(float) * ((BM * BK) + (BK * BN));
    constexpr int thread_registers = (TM * TIM) + (TN * TIN) + (TM * TN * WIM * WIN);
    constexpr int cta_registers = threads * thread_registers;

    constexpr int min_blocks = std::min({
        smem_bytes_sm / smem_bytes,
        regfile_size_sm / cta_registers,
        warps_sm / warps
    });
}
#endif