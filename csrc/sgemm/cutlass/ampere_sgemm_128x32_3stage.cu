#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include <cute/tensor.hpp>

namespace ampere_sgemm_128x32_3stage {

template <typename strideA, typename strideB, typename strideC,
          typename sALayout, typename sBLayout,
          typename ctaShape, typename copyPolicyA, typename copyPolicyB, typename mmaPolicy>
__global__ void ampere_sgemm_128x32_3stage(
    int m, int n, int k, float alpha, float beta,
    const float* __restrict__ A, strideA stride_A,
    const float* __restrict__ B, strideB stride_B,
    float* __restrict__ C, strideC stride_C,
    ctaShape cta_shape, sALayout sA_layout, sBLayout sB_layout,
    copyPolicyA copy_A, copyPolicyB copy_B, mmaPolicy tiled_mma
) {
    using namespace cute;
    using blockPipes = _2;  // num. of block buffers
    using pipeSize = _2;  // size of block in K-axis

    // opt for dynamic smem size here due to 48kb limitation on static alloc
    extern __shared__ float smem_buffer[];
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k), stride_A));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k), stride_B));
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(m, n), stride_C));

    auto coord = make_coord(blockIdx.y, blockIdx.x, _);
    auto gA = local_tile(mA, cta_shape, coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_shape, coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_shape, coord, Step<_1, _1, X>{});
    auto sA = make_tensor(make_smem_ptr(&smem_buffer[0]), sA_layout);
    auto sB = make_tensor(make_smem_ptr(&smem_buffer[cosize_v<sALayout>]), sB_layout);

    auto tA = copy_A.get_thread_slice(threadIdx.x);
    auto tAgA = tA.partition_S(gA);
    auto tAsA = tA.partition_D(sA);
    
    auto tB = copy_B.get_thread_slice(threadIdx.x);
    auto tBgB = tB.partition_S(gB);
    auto tBsB = tB.partition_D(sB);

    uint gmem_tile_idx = 0;
    uint gmem_tiles = size<2>(gA);
    constexpr uint smem_pipes = size<2>(sA);
    
    // prefetch for first (smem_pipes - 1) pipes
    for (uint i = 0; i < smem_pipes - 1; ++i) {
        copy(copy_A, tAgA(_,_,_,gmem_tile_idx), tAsA(_,_,_,i));
        copy(copy_B, tBgB(_,_,_,gmem_tile_idx), tBsB(_,_,_,i));
        cp_async_fence();
        --gmem_tiles;
        if (gmem_tiles) {
            ++gmem_tile_idx;
        }
    }

    auto tC = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = tC.partition_A(sA);
    auto tCsB = tC.partition_B(sB);
    auto tCgC = tC.partition_C(gC);
    // auto tCrA = make_fragment_like(tCsB(_,_,_,0));
    // auto tCrB = make_fragment_like(tCsB(_,_,_,0));
    auto tCrC = make_fragment_like(tCgC);
    fill(tCrC, 0.0);

#if 0
    if(thread0()) {
        print(tiled_mma.thrfrg_A(sA)); print("\n");
        print(tiled_mma.thrfrg_B(sB)); print("\n");
    }
#endif

#if 1
    auto tCrA = make_fragment_like(composition(tCsA(_,_,_,0), make_shape(_,_,blockPipes{})));
    auto tCrB = make_fragment_like(composition(tCsB(_,_,_,0), make_shape(_,_,blockPipes{})));

    int block_pipe = 0;
    cp_async_wait<smem_pipes - 2>();
    __syncthreads();
    // prefetch rmem_block = 0
    copy(tCsA(_,_,0,0), tCrA(_,_,block_pipe));  // (M, K, pipe)
    copy(tCsB(_,_,0,0), tCrB(_,_,block_pipe));  // (N, K, pipe)

    uint pipe_read = 0;
    uint pipe_write = smem_pipes - 1;
    constexpr uint rmem_blocks = size<2>(tCsA);
    const uint k_iters = gmem_tiles + (smem_pipes - 1);

    for (uint iter = 0;  iter < k_iters; ++iter) {
        cp_async_wait<smem_pipes - 2>();
        __syncthreads();

        CUTE_UNROLL
        for (uint block = 0; block < rmem_blocks - 1; ++block) {
            copy(tCsA(_,_,block+1,pipe_read), tCrA(_,_,block_pipe^1));  // TODO: call before gemm to interleave?
            gemm(tiled_mma, tCrA(_,_,block_pipe), tCrB(_,_,block_pipe), tCrC);
            copy(tCsB(_,_,block+1,pipe_read), tCrB(_,_,block_pipe^1));
            block_pipe ^= 1;
        }
        gemm(tiled_mma, tCrA(_,_,block_pipe), tCrB(_,_,block_pipe), tCrC);
        block_pipe ^= 1;

        if (iter < gmem_tiles) {
            copy(copy_A, tAgA(_,_,_,gmem_tile_idx), tAsA(_,_,_,pipe_write));    
            copy(copy_B, tBgB(_,_,_,gmem_tile_idx), tBsB(_,_,_,pipe_write));
        }
        cp_async_fence();
        
        pipe_write = pipe_read;
        pipe_read = (pipe_read + 1) % smem_pipes;

        if (iter != k_iters - 1) {
            copy(tCsA(_,_,0,pipe_read), tCrA(_,_,block_pipe));
            copy(tCsB(_,_,0,pipe_read), tCrB(_,_,block_pipe));
        }
        ++gmem_tile_idx;
    }
    axpby(alpha, tCrC, beta, tCgC);
#endif

/*
Vectorized 64-bit loads for B matrix in K-axis 
*/
#if 0
    auto tCrA = make_tensor<float>(
        make_layout(
            make_shape(_1{}, size<1>(tCsA), pipeSize{}, blockPipes{}),
            make_stride(_0{}, _1{}, size<1>(tCsA), size<1>(tCsA) * pipeSize{})
        )
    );  // (1, blockM, BlockPipeSizeK, RestBlockPipesK)
    auto tCrB = make_tensor<float>(
        make_layout(
            make_shape(_1{}, size<1>(tCsB), pipeSize{}, blockPipes{}),
            make_stride(_0{}, pipeSize{}, _1{}, size<1>(tCsB) * pipeSize{})
        )
    );  // (1, blockN, BlockPipeSizeK, RestBlockPipesK)

    auto s2r_copy_B = AutoVectorizingCopyWithAssumedAlignment<pipeSize{} * sizeof(float) * 8>{};
    auto tCsA_block = logical_divide(tCsA, make_shape(_,_,pipeSize{},_));  // (1, (CopyMVector, NumCopyMVectors), (BlockPipeSizeK, RestBlockPipesK), SmemPipes)
    auto tCsB_block = logical_divide(tCsB, make_shape(_,_,pipeSize{},_));  // (1, CopyN, (BlockPipeSizeK, RestBlockPipesK), SmemPipes)

    int block_pipe = 0;
    cp_async_wait<smem_pipes - 2>();
    __syncthreads();
    // prefetch rmem_block = 0
    copy(            tCsA_block(_,_,make_coord(_,0),0), tCrA(_,_,_,block_pipe));
    copy(s2r_copy_B, tCsB_block(_,_,make_coord(_,0),0), tCrB(_,_,_,block_pipe));

    uint pipe_read = 0;
    uint pipe_write = smem_pipes - 1;
    constexpr uint rmem_blocks = size<2>(tCsA);
    const uint k_iters = gmem_tiles + (smem_pipes - 1);

    for (uint iter = 0;  iter < k_iters; ++iter) {
        cp_async_wait<smem_pipes - 2>();
        __syncthreads();

        CUTE_UNROLL
        for (uint block = 0; block < (rmem_blocks / pipeSize{}) - 1; ++block) {
            copy(tCsA_block(_,_,make_coord(_,block+1),pipe_read), tCrA(_,_,_,block_pipe^1));
            gemm(tiled_mma, tCrA(_,_,_,block_pipe), tCrB(_,_,_,block_pipe), tCrC);
            copy(tCsB_block(_,_,make_coord(_,block+1),pipe_read), tCrB(_,_,_,block_pipe^1));
            block_pipe ^= 1;
        }
        gemm(tiled_mma, tCrA(_,_,_,block_pipe), tCrB(_,_,_,block_pipe), tCrC);
        block_pipe ^= 1;

        if (iter < gmem_tiles) {
            copy(copy_A, tAgA(_,_,_,gmem_tile_idx), tAsA(_,_,_,pipe_write));    
            copy(copy_B, tBgB(_,_,_,gmem_tile_idx), tBsB(_,_,_,pipe_write));
        }
        cp_async_fence();
        
        pipe_write = pipe_read;
        pipe_read = (pipe_read + 1) % smem_pipes;

        if (iter != k_iters - 1) {
            copy(tCsA_block(_,_,make_coord(_,0),pipe_read), tCrA(_,_,_,block_pipe));
            copy(tCsB_block(_,_,make_coord(_,0),pipe_read), tCrB(_,_,_,block_pipe));
        }
        ++gmem_tile_idx;
    }
    axpby(alpha, tCrC, beta, tCgC);
#endif
}

void nn(int m, int n, int k, float alpha,
        const float* A, int ldA, const float* B, int ldB, float beta, float* C,
        int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
    auto stride_A = make_stride(Int<1>{}, ldA);
    auto stride_B = make_stride(ldB, Int<1>{});
    auto stride_C = make_stride(Int<1>{}, ldC);

    // on SM86, this config. limits 1 CTA per SM--(((128 x 32 x 3) x 2) x 4) = 98304 bytes / CTA
    // TODO: experiment w/ a smaller K-tile or less pipes to increase occupancy?
    constexpr int n_pipes = 3;
    auto sA_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape), Int<n_pipes>{}));
    auto sB_layout = make_layout(
        make_shape(size<1>(cta_shape), size<2>(cta_shape), Int<n_pipes>{}),
        make_stride(size<2>(cta_shape), Int<1>{}, size<1>(cta_shape) * size<2>(cta_shape))
    );
    constexpr uint smem_size = (cosize_v<decltype(sA_layout)> + cosize_v<decltype(sB_layout)>) * sizeof(float);

    // at any given stage, tiled copy has 1 pipe of data in-flight
    // we use cp.async.cg instruction here b/c occupancy is 1 CTA / SM; no use for L1 caching (no space anyways, carveout is 98%)
    auto copy_A = make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{})),
        make_layout(make_shape(Int<4>{}, Int<1>{}))
    );
    auto copy_B = make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<4>{}), LayoutRight{})
    );

#if 1
    auto mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{},
        Layout<Shape<_16,_16>>{},
        Tile<
            Layout<Shape<_16,_4,_2>, Stride<_4,_1,_64>>,
            _128,
            _32
        >{}
    );
    auto kernel = ampere_sgemm_128x32_3stage<decltype(stride_A), decltype(stride_B), decltype(stride_C),
                                             decltype(sA_layout), decltype(sB_layout), decltype(cta_shape),
                                             decltype(copy_A), decltype(copy_B), decltype(mma)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 block_dim(size(mma));
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))), size(ceil_div(n, select<1>(cta_shape))));
    kernel<<<grid_dim, block_dim, smem_size, nullptr>>>(
        m, n, k, alpha, beta, A, stride_A, B, stride_B, C, stride_C, cta_shape, sA_layout, sB_layout, copy_A, copy_B, mma
    );
#endif

// "warp"-tiled MMA
#if 0
    // auto sB_layout_swizzled = composition(Swizzle<3,0,8>{}, sB_layout);  // fails w/ 128-bit loads
    auto sB_layout_swizzled = composition(Swizzle<3,2,6>{}, sB_layout);
    auto thr_layout = make_layout(
        make_layout(Shape<_4, _4>{}, Stride<_1, _32>{}),  // M: (LaneM, WarpM)
        make_layout(Shape<_8, _2>{}, Stride<_4, _128>{})  // N: (LaneN, WarpN)
    );
    auto mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{}, 
        thr_layout,
        Tile<
            Layout<Shape<_16,_8>, Stride<_8,_1>>,
            Layout<Shape<_16,_8>, Stride<_8,_1>>,
            Underscore
        >{}
    );
    auto kernel = ampere_sgemm_128x32_3stage<decltype(stride_A), decltype(stride_B), decltype(stride_C),
                                             decltype(sA_layout), decltype(sB_layout_swizzled), decltype(cta_shape),
                                             decltype(copy_A), decltype(copy_B), decltype(mma)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 block_dim(size(mma));
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))), size(ceil_div(n, select<1>(cta_shape))));
    kernel<<<grid_dim, block_dim, smem_size, nullptr>>>(
        m, n, k, alpha, beta, A, stride_A, B, stride_B, C, stride_C, cta_shape, sA_layout, sB_layout_swizzled, copy_A, copy_B, mma
    );
#endif
}

}  // namespace ampere_sgemm_128x32_3stage

void launch_ampere_sgemm_128x32_3stage(
    char transA, char transB, int m, int n, int k,
    float alpha, const float* A, int ldA,
    const float* B, int ldB, float beta,
    float* C, int ldC, cudaStream_t stream = 0
) {
    if (transA == 'N' && transB == 'N') {
        ampere_sgemm_128x32_3stage::nn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
    // using namespace cute;
    // auto lt = make_layout(make_shape(_128{}, _32{}), LayoutRight{});
    // auto swizzled = composition(Swizzle<3,2,6>{}, lt);
    // print_latex(swizzled); print("\n");
}