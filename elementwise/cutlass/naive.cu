#include <cute/tensor.hpp>

// (1:1) TV copy + compute
template <class AStride, class BStride, class CStride,
          class CTAShape, class ThreadLayout>
__global__ void elementwise_kernel_v1(
    int M, int N, float* A, AStride dA, float* B, BStride dB, float* C, CStride dC,
    CTAShape cta_shape, ThreadLayout thr_layout
) {
    using namespace cute;
    
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, N), dA);
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(M, N), dB);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    auto gA = local_tile(mA, cta_shape, make_coord(blockIdx.x, blockIdx.y));  // CTA-local views
    auto gB = local_tile(mB, cta_shape, make_coord(blockIdx.x, blockIdx.y));
    auto gC = local_tile(mC, cta_shape, make_coord(blockIdx.x, blockIdx.y));
    
    auto tgA = local_partition(gA, thr_layout, threadIdx.x);
    auto tgB = local_partition(gB, thr_layout, threadIdx.x);
    auto tgC = local_partition(gC, thr_layout, threadIdx.x);
    auto tgC_reg = make_tensor_like(tgC);

    CUTE_UNROLL
    for (uint i = 0; i < size(tgA); ++i) {
        tgC_reg(i) = tgA(i) + tgB(i);
    }
    copy(tgC_reg, tgC);

    #if 0
    if (thread(0, 0)) {
        print("  mA : "); print(mA); print("\n");
        print("  gA : "); print(gA); print("\n");
        print("  tgA : "); print(tgA); print("\n");
        print("  tgC : "); print(tgC); print("\n");
        print("  tgC_res : "); print(tgC_reg); print("\n");
        print("  gC : "); print(gC); print("\n");
    }
    #endif
}