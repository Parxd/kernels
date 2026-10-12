#include "cute/stride.hpp"
#include "cute/underscore.hpp"
#include <cute/tensor.hpp>

namespace sgemm_128x16_pipe {

template <typename strideA, typename strideB, typename strideC,
          typename sALayout, typename sBLayout,
          typename ctaShape, typename copyPolicyA, typename copyPolicyB, typename mmaPolicy>
__global__ void sgemm_128x16_pipe(
    int m, int n, int k, float alpha, float beta,
    const float* __restrict__ A, strideA stride_A,
    const float* __restrict__ B, strideB stride_B,
    float* __restrict__ C, strideC stride_C,
    ctaShape cta_shape, sALayout sA_layout, sBLayout sB_layout,
    copyPolicyA copy_A, copyPolicyB copy_B, mmaPolicy tiled_mma
) {
    using namespace cute;
    // TODO: ADD ASSERTIONS
    __shared__ float sA_buffer[cosize_v<sALayout>];
    __shared__ float sB_buffer[cosize_v<sBLayout>];
    auto blk_coord = make_coord(blockIdx.y, blockIdx.x, _);

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(m, k), stride_A);
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(n, k), stride_B);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(m, n), stride_C);
    auto gA = local_tile(mA, cta_shape, blk_coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_shape, blk_coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_shape, blk_coord, Step<_1, _1, X>{});
    auto sA = make_tensor(make_smem_ptr(sA_buffer), sA_layout);
    auto sB = make_tensor(make_smem_ptr(sB_buffer), sB_layout);

    auto tA = copy_A.get_thread_slice(threadIdx.x);
    auto tAgA = tA.partition_S(gA);
    auto tAsA = tA.partition_D(sA);
    auto tArA = make_fragment_like(tAsA);

    auto tB = copy_B.get_thread_slice(threadIdx.x);
    auto tBgB = tB.partition_S(gB);
    auto tBsB = tB.partition_D(sB);
    auto tBrB = make_fragment_like(tBsB);

    copy(copy_A, tAgA(_,_,_,0), tArA);
    copy(copy_B, tBgB(_,_,_,0), tBrB);
    
    auto tC = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = tC.partition_A(sA);
    auto tCsB = tC.partition_B(sB);
    auto tCgC = tC.partition_C(gC);
    auto tCrC = make_fragment_like(tCgC);

    uint k_iters = size<3>(tAgA);  // dynamic
    for (uint i = 0; i < k_iters; ++i) {
    /*
        hotloop:
        - rmem -> smem for k_iter
        - sync CTA
        - compute on smem for k_iter
        - gmem -> rmem for k_iter + 1
        - sync CTA
    */
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
        if (i + 1 < k_iters) {
            copy(copy_A, tAgA(_,_,_,i+1), tArA);
            copy(copy_B, tBgB(_,_,_,i+1), tBrB);
        }
        gemm(tCsA, tCsB, tCrC);
    }
    axpby(alpha, tCrC, beta, tCgC);
}

void nn(int m, int n, int k, float alpha,
              const float* A, int ldA, const float* B, int ldB, float beta, float* C,
              int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    auto problem_shape = make_shape(m, n, k);  // dynamic
    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<16>{});
    auto stride_A = make_stride(Int<1>{}, ldA);
    auto stride_B = make_stride(ldB, Int<1>{});
    auto stride_C = make_stride(Int<1>{}, ldC);

    auto sA_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape)));
    auto sB_layout = make_layout(make_shape(select<1>(cta_shape), select<2>(cta_shape)), LayoutRight{});

    TiledCopy copy_A = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{})),
        make_layout(make_shape(Int<4>{}, Int<1>{}))
    );
    TiledCopy copy_B = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        make_layout(make_shape(Int<64>{}, Int<4>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<4>{}))
    );
    TiledMMA mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})),
        Tile<
            // Layout<Shape<_16,_8>, Stride<_8,_1>>,
            Layout<Shape<_32,_4>, Stride<_4,_1>>,
            // _128,
            _128,
            _16
        >{}
    );

    dim3 cta_dim(size(mma));
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))),
                  size(ceil_div(n, select<1>(cta_shape))));
    sgemm_128x16_pipe<<<grid_dim, cta_dim, 0, stream>>>(
        m, n, k, alpha, beta,
        A, stride_A,
        B, stride_B,
        C, stride_C,
        cta_shape, sA_layout, sB_layout,
        copy_A, copy_B, mma
    );
}

}  // namespace sgemm_128x16_pipe

void launch_sgemm_128x16_pipe(
    char transA, char transB, int m, int n, int k,
    float alpha, const float* A, int ldA,
    const float* B, int ldB, float beta,
    float* C, int ldC, cudaStream_t stream = 0
) {
    if (transA == 'N' && transB == 'N') {
        sgemm_128x16_pipe::nn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
}   