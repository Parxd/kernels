import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def implicit_conv2d(
    v_act: cute.Tensor,
    filter: cute.Tensor,
    out: cute.Tensor,
    h_w_stride_pad: cute.IntTuple, # 4-element packed tuple: image height, width, stride, padding
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    tiler: cute.Shape,
    g2s_tiled_copy_A: cute.TiledCopy,
    g2s_tiled_copy_B: cute.TiledCopy,
    s2r_copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()
    
    @cute.struct
    class SharedStorage:
        a: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float16, cute.cosize(sA_layout)],
            16,
        ]
        b: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float16, cute.cosize(sB_layout)],
            16,
        ]
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sA = SharedStorage(storage).a.get_tensor(sA_layout)
    sB = SharedStorage(storage).b.get_tensor(sB_layout)

    GEMM_M = cute.size(v_act, [0])
    GEMM_N = cute.size(filter, [1])
    GEMM_K = cute.size(v_act, [1])
    mA = cute.zipped_divide(v_act, (tiler[0], tiler[2]))
    mA_pred = cute.zipped_divide(cute.make_identity_tensor(v_act.shape), (tiler[0], tiler[2]))
    mB = cute.zipped_divide(filter, (tiler[1], tiler[2]))
    gA = mA[(None, None), (bid_y, None)]  # TODO: re-assign CTAs for better L2 use
    gB = mB[(None, None), (bid_x, None)]
    gA_pred = mA_pred[(None, None), (bid_y, None)]
    ih, iw, stride, pad = h_w_stride_pad

    g2s_thr_copy_A = g2s_tiled_copy_A.get_slice(tid)
    g2s_thr_copy_B = g2s_tiled_copy_B.get_slice(tid)
    tAmA_pred = g2s_thr_copy_A.partition_S(gA_pred)
    tAgA = g2s_thr_copy_A.partition_S(gA)
    tAsA = g2s_thr_copy_A.partition_D(sA)
    tBgB = g2s_thr_copy_B.partition_S(gB)
    tBsB = g2s_thr_copy_B.partition_D(sB)

    tApA = cute.make_rmem_tensor(
        layout_or_shape=cute.make_layout(
            (tAgA.shape[0][1], tAgA.shape[1], tAgA.shape[2], cute.size(tAgA, [3]))
        ),
        dtype=cutlass.Boolean
    )
    for frg_x in cutlass.range_constexpr(tApA.shape[0]):
        for rest_m in cutlass.range_constexpr(tApA.shape[1]):
            # for rest_n in cutlass.range_constexpr(tApA.shape[2]):
            for rest_k in cutlass.range_constexpr(tApA.shape[3]):
                (_, p, q), (r, s, _) = tAmA_pred[frg_x, rest_m, 0, rest_k] 
                h, w = p * stride + r, q * stride + s
                tApA[frg_x, rest_m, 0, rest_k] =  h >= pad and w >= pad and h < ih - pad and w < iw - pad


@cute.jit
def entry(
    activations: cute.Tensor,
    filter: cute.Tensor,
    out: cute.Tensor,
    stride: cutlass.Constexpr = 2,
    pad: cutlass.Constexpr = 1,
    num_threads: cutlass.Constexpr = 256
):
    batch_size, height, width, in_channels = activations.shape
    out_channels, filter_height, filter_width, _ = filter.shape
    _, out_height, out_width, _ = out.shape

    # ------ Tiler Config ------
    TILE_P = 16
    TILE_Q = 16
    tiler_m = (1, TILE_P, TILE_Q)
    tiler_n = out_channels
    tiler_k = (1, 1, in_channels)
    num_stages = 3
    # --------------------------

    v_act_lt = cute.make_layout(
        shape=(
            (batch_size, out_height, out_width),
            (filter_height, filter_width, in_channels)
        ),
        stride=(
            ((height + pad * 2) * (width + pad * 2) * in_channels, (width + pad * 2) * in_channels, in_channels),
            (in_channels * filter_width, in_channels, 1)
        )
    )
    v_act = cute.domain_offset(
        ((0, -pad, -pad), (0, 0, 0)),  # point iterator to negative OOB memory, will use predicate to avoid segfault
        cute.make_tensor(iterator=activations.iterator.align(16), layout=v_act_lt)
    )
    grouped_filter = cute.group_modes(filter, 1, 4)
    sA_layout = cute.make_ordered_layout((cute.size(tiler_m), cute.size(tiler_k), num_stages), order=(1, 0, 2))
    sB_layout = cute.make_ordered_layout((cute.size(tiler_n), cute.size(tiler_k), num_stages), order=(1, 0, 2))
    
    # max 128-bit load w/ fp16 = 8x fp16
    TV_SIZE = 128 // 16
    THR_K = cute.size(tiler_k) // TV_SIZE

    tA = cute.make_layout(
        shape=((1, num_threads // THR_K // TILE_Q, TILE_Q), (1, 1, THR_K)),
        stride=((0, TILE_Q * THR_K, THR_K), (0, 0, 1))
    )

    vA = cute.make_layout(((1, 1, 1), (1, 1, TV_SIZE)), stride=((0, 0, 0), (0, 0, 1)))
    tB = cute.make_layout(
        shape=(num_threads // THR_K, (1, 1, THR_K)),
        stride=(THR_K, (0, 0, 1))
    )
    vB = cute.make_layout((1, (1, 1, TV_SIZE)), stride=(0, (0, 0, 1)))
    g2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float16,
        num_bits_per_copy=TV_SIZE * cutlass.Float16.width
    )
    g2s_copy_A = cute.make_tiled_copy_tv(g2s_copy_atom, tA, vA)
    g2s_copy_B = cute.make_tiled_copy_tv(g2s_copy_atom, tB, vB)

    s2r_copy_atom = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False,
            num_matrices=1
        ),
        copy_internal_type=cutlass.Float16
    )
    mma_atom = cute.make_mma_atom(
        op=cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16)
        )
    )
    tiled_mma = cute.make_tiled_mma(mma_atom)
    
    grid_dim = cute.ceil_div(
        (out_channels, batch_size * out_height * out_width),
        (cute.size(tiler_n), cute.size(tiler_m))
    )
    implicit_conv2d(
        v_act,
        grouped_filter,
        out,
        (height, width, stride, pad),
        sA_layout, sB_layout,
        (tiler_m, tiler_n, tiler_k),
        g2s_copy_A, g2s_copy_B,
        s2r_copy_atom,
        tiled_mma
    ).launch(
        grid=grid_dim,
        block=(num_threads, 1, 1)
    )


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

    test_activations = torch.randn((N, H, W, C), dtype=torch.float16).to('cuda')
    test_filter = torch.randn((K, R, S, C), dtype=torch.float16).to('cuda')
    test_out = torch.empty((N, P, Q, K), dtype=torch.float16).to('cuda')
    entry(from_dlpack(test_activations), from_dlpack(test_filter), from_dlpack(test_out))

if __name__ == "__main__":
    main()
