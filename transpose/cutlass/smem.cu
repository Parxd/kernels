#include <cute/tensor.hpp>

template <class mATensor, class mBTensor, class sALayout, class sBLayout>
__global__ void __launch_bounds__(256, 1)
smem_transpose(mATensor mA, mBTensor mB, sALayout sA_layout, sBLayout sB_layout) {
    using namespace cute;
    using thr_lt = Layout<Shape<_8,_32>,Stride<_32,_1>>;

    __shared__ float smem_buffer[cosize_v<sALayout>];
    auto gA = mA(make_coord(_,_), blockIdx.y, blockIdx.x);
    auto gB = mB(make_coord(_,_), blockIdx.x, blockIdx.y);
    auto sA = make_tensor(make_smem_ptr(smem_buffer), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem_buffer), sB_layout);

    auto thr_layout = thr_lt{};
    auto tAgA = local_partition(gA, thr_layout, threadIdx.x);
    auto tAsA = local_partition(sA, thr_layout, threadIdx.x);

    copy(tAgA, tAsA);
    __syncthreads();

    auto tBsB = local_partition(sB, thr_layout, threadIdx.x);
    auto tBgB = local_partition(gB, thr_layout, threadIdx.x);

    copy(tBsB, tBgB);
    // __syncthreads();
}

void launch_smem_transpose(int m, int n, float* A, float* B) {
    using namespace cute;
    using thr_lt = Layout<Shape<_8,_32>,Stride<_32,_1>>;
    
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(m, n), make_stride(n, _1{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(m, n), make_stride(n, _1{}));
    auto tile_shape = make_shape(_64{}, _64{});
    auto tile_lt_S = make_layout(tile_shape, LayoutRight{});
    auto tile_lt_D = composition(tile_lt_S, make_layout(tile_shape, LayoutRight{}));  // LayoutRight layout composed with itself -> LayoutLeft

    auto tensor_S = tiled_divide(gA, tile_shape);
    auto tensor_D = tiled_divide(gB, tile_shape);

    dim3 gridDim(size<1>(tensor_S), size<2>(tensor_S));
    dim3 blockDim(size(thr_lt{}));
    smem_transpose<<<gridDim, blockDim, 0, nullptr>>>(tensor_S, tensor_D, tile_lt_S, tile_lt_D);
}