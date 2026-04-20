#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

using namespace cute;

template <class Layout, class TiledCopy>
__global__ void test_kernel(int* in, int* out, Layout layout, TiledCopy tiled_copy) {
    using swizzle = Swizzle<5,1,5>;
    auto smem_layout = make_layout(make_shape(_32{}, _32{}), LayoutRight{});
    auto smem_layout_swizzled = composition(swizzle{}, smem_layout);

    if(thread0()) {
        print_layout(smem_layout_swizzled); print("\n");
        // print(swizzle::yyy_msk{});
    }

    // __shared__ int sA_buffer[cute::cosize_v<decltype(smem_layout)>];
    // auto mA = make_tensor(make_gmem_ptr(in), layout);
    // auto mB = make_tensor(make_gmem_ptr(out), layout);
    // auto gA = mA(make_coord(_,_), blockIdx.y, blockIdx.x);
    // auto gB = mB(make_coord(_,_), blockIdx.y, blockIdx.x);
    // auto sA = make_tensor(make_smem_ptr(sA_buffer), smem_layout);
    
    // auto thr_view = tiled_copy.get_thread_slice(threadIdx.x);
    // auto copy_gA = thr_view.partition_S(gA);
    // auto copy_sA = thr_view.partition_D(sA);
    // auto copy_gB = thr_view.partition_D(gB);

    // copy(tiled_copy, copy_gA, copy_sA);
    // copy(tiled_copy, copy_sA, copy_gB);
    // __syncthreads();

    // if(thread0()) {
        // print(decltype(tiled_copy)::tidfrg_S(gA)); print("\n");
        // print(decltype(tiled_copy)::tidfrg_D(sA)); print("\n");
    // }
}

int main() {
    auto m = 1024;
    auto n = 1024;

    auto in_H = thrust::host_vector<int>(m * n);
    auto out_H = thrust::host_vector<int>(m * n);
    std::iota(in_H.begin(), in_H.end(), 0.0f);
    thrust::device_vector<int> in_D = in_H;
    thrust::device_vector<int> out_D(m * n);

    auto layout = make_layout(make_shape(m, n), LayoutRight{});
    auto tile_shape = make_shape(_32{},_32{});
    auto div = tiled_divide(layout, tile_shape);

    auto thr_layout = make_layout(make_shape(_16{}, _16{}), LayoutRight{});
    auto val_layout = make_layout(make_shape(_1{}, _4{}));
    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, thr_layout, val_layout);

    dim3 gridDim(size<1>(div), size<2>(div));
    dim3 blockDim(size(thr_layout));
    test_kernel<<<gridDim, blockDim, 0, nullptr>>>(in_D.data().get(), out_D.data().get(), div, tiled_copy);

    out_H = out_D;

    // auto ref = thrust::host_vector<int>(m * n);
    // ref = in_H;
    // for (int i = 0; i < out_H.size(); ++i) {
    //     if (ref[i] != out_H[i]) {
    //         std::cout << "mismatch at (" << i / n << "," << i % n << ")" << "\n";
    //     }
    // }

    return 0;
}