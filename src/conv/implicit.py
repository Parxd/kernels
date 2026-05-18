import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def implicit_conv(activations: cute.Tensor, filter: cute.Tensor, out: cute.Tensor,
                  N, H, W, C, R, S, P, Q, K, stride, pad):
    tid, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()


@cute.jit
def entry(activations: cute.Tensor, filter: cute.Tensor, out: cute.Tensor,
          stride: cutlass.Constexpr = 2, pad: cutlass.Constexpr = 1,
          num_threads: cutlass.Constexpr = 256):
    
    batch_size, height, width, in_channels = activations.shape
    out_channels, filter_height, filter_width, _ = filter.shape
    _, _, out_height, out_width = out.shape

    # ------ Tiler Config ------
    TILE_P = 16
    TILE_Q = 16
    tiler_m = (1, TILE_P, TILE_Q)
    tiler_n = out_channels
    tiler_k = (1, 1, in_channels)
    num_stages = 3
    # --------------------------

    virtual_act_lt = cute.make_layout(
        shape=(
            (batch_size, out_height, out_width),
            (filter_height, filter_width, in_channels)
        ),
        stride=(
            ((height + pad * 2) * (width + pad * 2) * in_channels, (width + pad * 2) * in_channels, in_channels),
            (in_channels * filter_width, in_channels, 1)
        )
    )
    padded_act = cute.domain_offset(
        ((0, -pad, -pad), (0, 0, 0)),
        cute.make_tensor(iterator=activations.iterator, layout=virtual_act_lt)
    )
    grouped_filter = cute.group_modes(filter, 1, 4)
    mA = cute.zipped_divide(padded_act, (tiler_m, tiler_k))
    mB = cute.zipped_divide(grouped_filter, (tiler_n, tiler_k))
    
    GEMM_M = cute.size(virtual_act_lt, [0])
    GEMM_N = cute.size(filter, [1])
    GEMM_K = cute.size(virtual_act_lt, [1])

    sA_layout = cute.make_ordered_layout((cute.size(tiler_m), cute.size(tiler_k), num_stages), order=(1, 0, 2))
    sB_layout = cute.make_ordered_layout((cute.size(tiler_n), cute.size(tiler_k), num_stages), order=(1, 0, 2))
    
    # max 128-bit load w/ fp16 = 8x fp16
    TV_SIZE = 128 // 16
    THR_K = cute.size(tiler_k) // TV_SIZE

    # TV mappings
    tA = cute.make_layout(
        shape=((1, num_threads // THR_K // TILE_Q, TILE_Q), (1, 1, THR_K)),
        stride=((0, TILE_Q * THR_K, THR_K), (0, 0, 1))
    )
    tB = cute.make_layout(
        shape=(num_threads // THR_K, (1, 1, THR_K)),
        stride=(THR_K, (0, 0, 1))
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float16,
        num_bits_per_copy=TV_SIZE * cutlass.Float16.width
    )
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, tA, cute.make_layout(((1, 1, 1), (1, 1, TV_SIZE))))
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, tB, cute.make_layout((1, 1, TV_SIZE)))

    sA = mA[((None, None), ((0, 0, 0), (0, 0, 0)))]
    sB = mB[((None, None), (0, 0))]

    print(tiled_copy_A)


def main():
    STRIDE, PAD = 2, 1
    N, H, W, C = 1, 320, 320, 32
    K, R, S = 64, 3, 3
    P = (H + 2 * PAD - R) // STRIDE + 1
    Q = (W + 2 * PAD - S) // STRIDE + 1

    # activations_nhwc = torch.rand((N, H, W, C), dtype=torch.float16, device='cuda')
    # filter = torch.rand((K, R, S, C), dtype=torch.float16, device='cuda')
    # ref = torch.conv2d(
        # activations_nhwc.permute(0, 3, 1, 2).contiguous(),  # NHWC -> NCHW
        # filter
        # pad=1
    # ).permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    # print(ref.shape)

    test_activations = torch.linspace(0, (N * H * W * C), (N * H * W * C), dtype=torch.int32).reshape(N, H, W, C)
    test_filter = torch.linspace(0, (K * R * S * C), (K * R * S * C), dtype=torch.int32).reshape(K, R, S, C)
    test_out = torch.empty((N, K, P, Q))
    # test_activations = torch.linspace(0, (N * H * W * C), (N * H * W * C), dtype=torch.int32).reshape(N, H, W, C).to('cuda')
    # test_filter = torch.linspace(0, (K * R * S * C), (K * R * S * C), dtype=torch.int32).reshape(K, R * S * C).to('cuda')
    # test_out = torch.empty((N * K * P * Q)).to('cuda')
    entry(from_dlpack(test_activations), from_dlpack(test_filter), from_dlpack(test_out))

if __name__ == "__main__":
    main()
