#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utils.h"
#include "cutlass/naive.cu"
#include "cutlass/vectorize.cu"
#include "cutlass/predicated.cu"

int main(int argc, char* argv[]) {
    int kernel = 0;
    int M = 256;
    int N = 256;
    bool time = false;
    bool check = false;

    if (argc > 1) kernel = std::atoi(argv[1]);
    if (argc > 2) M = std::atoi(argv[2]);
    if (argc > 3) N = std::atoi(argv[3]);
    if (argc > 4) time = bool(std::atoi(argv[4]));
    if (argc > 5) check = bool(std::atoi(argv[5]));
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0]
                  << " [kernel: int=0-2] [M: int=256] [N: int=256] [time: bool=0] [verify: bool=0]\n";
        return 0;
    }

    std::cout << "[Elementwise]: Running kernel=" << kernel
              << " with M=" << M
              << ", N=" << N
              << " (time=" << time
              << ", verify=" << check
              << ")\n";

    using namespace cute;
    using dtype = float;

    auto hA = thrust::host_vector<dtype>(M * N);
    auto hB = thrust::host_vector<dtype>(M * N);
    auto hC = thrust::host_vector<dtype>(M * N);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < M * N; ++i) {
        hA[i] = static_cast<dtype>(std::rand()) / static_cast<dtype>(RAND_MAX);
        hB[i] = static_cast<dtype>(std::rand()) / static_cast<dtype>(RAND_MAX);
    }
    auto dA = thrust::device_vector<dtype>(M * N);
    auto dB = thrust::device_vector<dtype>(M * N);
    auto dC = thrust::device_vector<dtype>(M * N, 0);
    dA = hA; dB = hB;

    auto str_A = make_stride(N, Int<1>{});
    auto str_B = make_stride(N, Int<1>{});
    auto str_C = make_stride(N, Int<1>{});

    auto thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
    // ensure that non-1 mode here aligns with mode to be vectorized
    auto val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}), LayoutRight{});

    // rather than forcing the block shape to be the raked prod. of thr + val layouts, define a custom shape that one CTA
    // should handle, then just let TiledCopy's partition methods automatically replicate the thread layouts across src. tensor
    auto cta_shape = product_each(shape(raked_product(thr_layout, val_layout)));
    
    // hence, opt for a statically defined block shape as follows, and set value layout to be the minimum number of elements that
    // fit in the CopyTrait width -- i.e. (4, 1) for fp32 loads using uint128_t
    auto block_shape = make_shape(Int<32>{}, Int<32>{});
    auto tiled = tiled_divide(make_layout(make_shape(M, N), str_A), block_shape);

    using copy_trait = Copy_Traits<UniversalCopy<uint128_t>>;
    using copy_atom = Copy_Atom<copy_trait, dtype>;
    auto tiled_copy = make_tiled_copy(copy_atom{}, thr_layout, val_layout);

    dim3 block_dim(size(thr_layout));
    dim3 grid_dim(size<1>(tiled), size<2>(tiled));
    if (kernel == 0) {
        assert(!(M % size<0>(block_shape)));
        assert(!(N % size<1>(block_shape)));
        
        elementwise_kernel_v1<<<grid_dim, block_dim, 0, nullptr>>>(
            M, N, dA.data().get(), str_A, dB.data().get(), str_B, dC.data().get(), str_C,
            block_shape, thr_layout
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if (kernel == 1) {
        assert(!(M % size<0>(block_shape)));
        assert(!(N % size<1>(block_shape)));

        elementwise_kernel_v2<<<grid_dim, block_dim, 0, nullptr>>>(
            M, N, dA.data().get(), str_A, dB.data().get(), str_B, dC.data().get(), str_C,
            block_shape, tiled_copy
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if (kernel == 2) {
        assert(!(M % size<1>(val_layout)));
        assert(!(N % size<1>(val_layout)));

        elementwise_kernel_v3<<<grid_dim, block_dim, 0, nullptr>>>(
            M, N, dA.data().get(), str_A, dB.data().get(), str_B, dC.data().get(), str_C,
            block_shape, tiled_copy
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    hC = dC;

    if (check) {
        bool failed = false;
        float delta = 0.001;
        std::cout << "[Elementwise]: Verifying results..." << "\n";
        for (uint i = 0; i < hA.size(); ++i) {
            if (std::abs(hC[i] - (hA[i] + hB[i])) > delta) {
                std::cerr << "[Elementwise]: Mismatch at M=" << i / N << ", N=" << i % N << "\n";
                failed = true;
            }
        }
        if (failed) return 1;
        else std::cout << "[Elementwise]: All elements match for delta=" << delta << "\n";
    }
    else {
        std::cout << "[Elementwise]: Verification off" << "\n";
    }

    // ------ for reference ------
    #if 0
    auto src_vec = std::vector<int>(16, 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);
    auto dest_vec = std::vector<int>(16, 0);
    auto src_tensor = make_tensor(src_vec.data(), make_shape(Int<4>{}, Int<4>{}), LayoutRight{});
    auto dest_tensor = make_tensor(dest_vec.data(), make_shape(Int<4>{}, Int<4>{}), LayoutRight{});

    using trait = Copy_Traits<UniversalCopy<uint128_t>>;  // LOGICAL type can differ from copy instruction WIDTH type
    using atom = Copy_Atom<trait, int>;
    auto threads = make_layout(make_shape(Int<1>{}, Int<4>{})); 
    auto values = make_layout(make_shape(Int<4>{}, Int<1>{}), LayoutRight{});  // value layout must match requirement in CopyAtom
    // raked prod. of thr + val layouts must be compatible with the source tensor
    auto tiled_copy = make_tiled_copy(copy_atom{}, threads, values);
    
    using TiledCopy = decltype(tiled_copy);
    auto thr_tensor = make_tensor(src_tensor.data(), TiledCopy::tidfrg_S(src_tensor.layout()));
    print("thr tensor: "); print_tensor(thr_tensor); print("\n");
    print("thr0 view: "); print_tensor(thr_tensor(0, _, repeat<rank_v<decltype(src_tensor)>>(_))); print("\n");
    print("tiler mn: "); print_layout(raked_product(threads, values)); print("\n");
    // print(repeat<rank_v<decltype(src_tensor)>>(_));  // IntTuple with rank of src_tensor, of _ elements
    
    // auto thr_copy = tiled_copy.get_thread_slice(0);
    // auto src_part = thr_copy.partition_S(src_tensor);
    // auto dest_part = thr_copy.partition_D(dest_tensor);
    // copy(tiled_copy, src_part, dest_part);
    // print_tensor(dest_tensor);
    #endif

    return 0;
}