#include <cute/tensor.hpp>

// (1:4) vectorized TV copy + compute
template <class AStride, class BStride, class CStride,
          class CTAShape, class TiledCopy>
__global__ void elementwise_kernel_v2(
    int M, int N, float* A, AStride dA, float* B, BStride dB, float* C, CStride dC,
    CTAShape cta_shape, TiledCopy copy_policy
) {
    using namespace cute;
    
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, N), dA);
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(M, N), dB);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    auto gA = local_tile(mA, cta_shape, make_coord(blockIdx.x, blockIdx.y));  // CTA-local
    auto gB = local_tile(mB, cta_shape, make_coord(blockIdx.x, blockIdx.y));
    auto gC = local_tile(mC, cta_shape, make_coord(blockIdx.x, blockIdx.y));

    auto thr_copy = copy_policy.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tArA = make_fragment_like(tAgA);
    auto tBgB = thr_copy.partition_S(gB);
    auto tBrB = make_fragment_like(tBgB);
    auto tCgC = thr_copy.partition_D(gC);
    auto tCrC = make_fragment_like(tCgC);

    copy(copy_policy, tAgA, tArA);      
    copy(copy_policy, tBgB, tBrB);

    #if 0
    if(thread(0,0)) {
        print("  gA: "); print(gA); print("\n");
        print("  gB: "); print(gB); print("\n");
        print("  tAgA: "); print(tAgA); print("\n");
        print("  tArA: "); print(tArA); print("\n");
    }
    #endif

    transform(tArA, tBrB, tCrC, [](auto i, auto j) { return i + j; });
    copy(copy_policy, tCrC, tCgC);
}
