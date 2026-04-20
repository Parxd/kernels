#include <cute/tensor.hpp>

// (1:4) vectorized TV copy + compute w/ predication
template <class AStride, class BStride, class CStride,
          class CTAShape, class TiledCopy>
__global__ void elementwise_kernel_v3(
    int M, int N, float* A, AStride dA, float* B, BStride dB, float* C, CStride dC,
    CTAShape cta_shape, TiledCopy copy_policy
) {
    using namespace cute;
    
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, N), dA);
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(M, N), dB);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto identity = make_identity_tensor(shape(mA));
    // note: lazily builds a static register-backed bool tensor view from a dynamic shape !
    auto predicate = cute::lazy::transform(identity, [&](auto i) { return elem_less(i, shape(mA)); });

    auto block = make_coord(blockIdx.x, blockIdx.y);
    auto gA = local_tile(mA, cta_shape, block);
    auto gB = local_tile(mB, cta_shape, block);
    auto gC = local_tile(mC, cta_shape, block);
    auto block_pred = local_tile(predicate, cta_shape, block);

    auto thr_copy = copy_policy.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto tCgC = thr_copy.partition_D(gC);
    auto thr_pred = thr_copy.partition_S(block_pred);

    auto tArA = make_fragment_like(tAgA);
    auto tBrB = make_fragment_like(tBgB);
    auto tCrC = make_fragment_like(tCgC);

    copy_if(copy_policy, thr_pred, tAgA, tArA);
    copy_if(copy_policy, thr_pred, tBgB, tBrB);
    transform(tArA, tBrB, tCrC, [](auto i, auto j) { return i + j; });

    #if 0
    if (thread0()) {
        print("  Predicate: "); print(predicate); print("\n");
        print("  g_pred: "); print_tensor(block_pred); print("\n");
        print("  t_pred: "); print_tensor(thr_pred); print("\n");
        print("  tArA: "); print_tensor(tArA); print("\n");
    }
    #endif

    copy_if(copy_policy, thr_pred, tCrC, tCgC);
}
