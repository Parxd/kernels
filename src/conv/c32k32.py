import cutlass
import cutlass.cute as cute


@cute.kernel
def implicit_conv2d(
    v_act: cute.Tensor,
    filter: cute.Tensor,
    out: cute.Tensor,
    ih_iw_stride_pad: cute.IntTuple,  # 4-element packed tuple: image height, width, stride, padding
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiler: cute.Shape,
    g2s_tiled_copy_A: cute.TiledCopy,
    g2s_tiled_copy_B: cute.TiledCopy,
    s2r_copy_atom_A: cute.CopyAtom,
    s2r_copy_atom_B: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()
    GEMM_M = cute.size(v_act, [0])
    GEMM_N = cute.size(filter, [1])
    GEMM_K = cute.size(v_act, [1])
    ih, iw, stride, pad = ih_iw_stride_pad
    
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
    
    mA = cute.zipped_divide(v_act, (tiler[0], tiler[2]))
    mA_pred = cute.zipped_divide(cute.make_identity_tensor(v_act.shape), (tiler[0], tiler[2]))
    mB = cute.zipped_divide(filter, (tiler[1], tiler[2]))
    mB_pred = cute.zipped_divide(cute.make_identity_tensor(filter.shape), (tiler[1], tiler[2]))
    mC = cute.zipped_divide(out, (tiler[0], tiler[1]))
    mC_pred = cute.zipped_divide(cute.make_identity_tensor(out.shape), (tiler[0], tiler[1]))
    
    gA = mA[(None, None), (bid_y, None)]  # TODO: re-assign CTAs for better L2 use
    gA_pred = mA_pred[(None, None), (bid_y, None)]
    gB = mB[(None, None), (bid_x, None)]
    gB_pred = mB_pred[(None, None), (bid_x, None)]
    gC = mC[(None, None), (bid_y, bid_x)]
    gC_pred = mC_pred[(None, None), (bid_y, bid_x)]
    TILES = cute.size(gA, [2])
    S_PIPES = sA.shape[2]
    R_PIPES = 2

    g2s_thr_copy_A = g2s_tiled_copy_A.get_slice(tid)
    g2s_thr_copy_B = g2s_tiled_copy_B.get_slice(tid)
    tAmA_pred = g2s_thr_copy_A.partition_S(gA_pred)
    tBmB_pred = g2s_thr_copy_B.partition_S(gB_pred)
    tAgA = g2s_thr_copy_A.partition_S(gA)
    tAsA = g2s_thr_copy_A.partition_D(sA)
    tBgB = g2s_thr_copy_B.partition_S(gB)
    tBsB = g2s_thr_copy_B.partition_D(sB)
    tAsA.fill(0.0)
    tBsB.fill(0.0)

    tApA = cute.make_rmem_tensor(
        layout_or_shape=(tAgA.shape[0][1], tAgA.shape[1], tAgA.shape[2], tAgA.shape[3]),
        dtype=cutlass.Boolean
    )
    for frg_x in cutlass.range_constexpr(tApA.shape[0]):
        for rest_m in cutlass.range_constexpr(tApA.shape[1]):
            for rest_n in cutlass.range_constexpr(tApA.shape[2]):
                for rest_k in range(cute.size(tApA, [3])):
                    (_, p, q), (r, s, c) = tAmA_pred[frg_x, rest_m, rest_n, rest_k] 
                    h, w = p * stride + r, q * stride + s
                    tApA[frg_x, rest_m, rest_n, rest_k] = \
                        p < v_act.shape[0][1] and q < v_act.shape[0][2] and c < v_act.shape[1][2] and \
                        h >= pad and w >= pad and h < ih + pad and w < iw + pad
    tBpB = cute.make_rmem_tensor(
        layout_or_shape=(tBgB.shape[0][1], tBgB.shape[1], tBgB.shape[2], tBgB.shape[3]),
        dtype=cutlass.Boolean
    )
    for frg_x in cutlass.range_constexpr(tBpB.shape[0]):
        for rest_m in cutlass.range_constexpr(tBpB.shape[1]):
            for rest_n in cutlass.range_constexpr(tBpB.shape[2]):
                for rest_k in range(cute.size(tBpB, [3])):
                    (k, (_, _, c)) = tBmB_pred[frg_x, rest_m, rest_n, rest_k]
                    tBpB[frg_x, rest_m, rest_n, rest_k] = k < filter.shape[0] and c < v_act.shape[1][2]

    s2r_tiled_copy_A = cute.make_tiled_copy_A(s2r_copy_atom_A, tiled_mma)
    s2r_tiled_copy_B = cute.make_tiled_copy_B(s2r_copy_atom_B, tiled_mma)
    thr_mma = tiled_mma.get_slice(tid)
    s2r_thr_copy_A = s2r_tiled_copy_A.get_slice(tid)
    s2r_thr_copy_B = s2r_tiled_copy_B.get_slice(tid)

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCmC_pred = thr_mma.partition_C(gC_pred)
    (_, mma_atom_m), mma_tile_m, mma_tile_k = tCgC.shape

    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrA = cute.composition(tCrA, (None, None, R_PIPES))
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrB = cute.composition(tCrB, (None, None, R_PIPES))
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC_reshape = cute.make_rmem_tensor(
        cute.make_layout(
            (((2, 2, 2), 1), mma_atom_m, mma_tile_m),
            stride=(((1, 2, 4), 0), 8, 8 * mma_atom_m)
        ),
        dtype=cutlass.Float16
    )
    tCgC_reshape = cute.make_tensor(
        tCgC.iterator.align(16),  # 8x fp16 = 16-byte align
        cute.make_layout(
            ((8, 1), mma_atom_m, mma_tile_m),
            stride=((1, 0), tCgC.stride[0][1], tCgC.stride[1])
        )
    )
    tCpC = cute.make_rmem_tensor(
        layout_or_shape=cute.make_layout((1, mma_atom_m, mma_tile_m)),
        dtype=cutlass.Boolean
    )
    for atom_m in cutlass.range_constexpr(mma_atom_m):
        for tile_m in cutlass.range_constexpr(mma_tile_m):
            ((_, p, q), k) = tCmC_pred[(0, atom_m), tile_m, 0]
            tCpC[0, atom_m, tile_m] = p < v_act.shape[0][1] and q < v_act.shape[0][2] and k < filter.shape[0]
    tCsA_copy = s2r_thr_copy_A.partition_S(sA)
    tCrA_copy = s2r_thr_copy_A.retile(tCrA)
    tCsB_copy = s2r_thr_copy_B.partition_S(sB)
    tCrB_copy = s2r_thr_copy_B.retile(tCrB)
    tCrC.fill(0.0)

    tile_idx = 0
    for pipe in cutlass.range_constexpr(S_PIPES - 1):
        cute.copy(g2s_thr_copy_A, tAgA[None, None, None, tile_idx], tAsA[None, None, None, pipe], pred=tApA[None, None, None, tile_idx])
        cute.copy(g2s_thr_copy_B, tBgB[None, None, None, tile_idx], tBsB[None, None, None, pipe], pred=tBpB[None, None, None, tile_idx])
        cute.arch.cp_async_commit_group()
        TILES -= 1
        if TILES:
            tile_idx += 1
    
    block_idx = 0
    cute.arch.cp_async_wait_group(S_PIPES - 2)
    cute.arch.sync_threads()
    cute.copy(s2r_tiled_copy_A, tCsA_copy[None, None, 0, 0], tCrA_copy[None, None, block_idx])
    cute.copy(s2r_tiled_copy_B, tCsB_copy[None, None, 0, 0], tCrB_copy[None, None, block_idx])

    pipe_r, pipe_w = 0, S_PIPES - 1
    BLOCKS = cute.size(tCsA, [2])
    K_ITERS = TILES + (S_PIPES - 1)

    for k_iter in range(K_ITERS):
        for block in cutlass.range_constexpr(BLOCKS - 1):
            cute.copy(s2r_tiled_copy_A, tCsA_copy[None, None, block + 1, pipe_r], tCrA_copy[None, None, block_idx ^ 1])
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, block_idx], tCrB[None, None, block_idx], tCrC)
            cute.copy(s2r_tiled_copy_B, tCsB_copy[None, None, block + 1, pipe_r], tCrB_copy[None, None, block_idx ^ 1])
            block_idx ^= 1
        cute.gemm(tiled_mma, tCrC, tCrA[None, None, block_idx], tCrB[None, None, block_idx], tCrC)
        block_idx ^= 1

        if k_iter < TILES:
            cute.copy(g2s_tiled_copy_A, tAgA[None, None, None, tile_idx], tAsA[None, None, None, pipe_w], pred=tApA[None, None, None, tile_idx])
            cute.copy(g2s_tiled_copy_B, tBgB[None, None, None, tile_idx], tBsB[None, None, None, pipe_w], pred=tBpB[None, None, None, tile_idx])
        cute.arch.cp_async_commit_group()

        pipe_w = pipe_r
        pipe_r = (pipe_r + 1) % S_PIPES

        if k_iter != K_ITERS - 1:
            cute.arch.cp_async_wait_group(S_PIPES - 2)
            cute.arch.sync_threads()
            cute.copy(s2r_tiled_copy_A, tCsA_copy[None, None, 0, pipe_r], tCrA_copy[None, None, block_idx])
            cute.copy(s2r_tiled_copy_B, tCsB_copy[None, None, 0, pipe_r], tCrB_copy[None, None, block_idx])
        tile_idx += 1
    
    # reorder register fragments from MMA to vectorize r2g stores 
    r2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    for atom_m in cutlass.range_constexpr(mma_atom_m):
        for tile_m in cutlass.range_constexpr(mma_tile_m):
            for tile_k in cutlass.range_constexpr(mma_tile_k):
                cute.copy(r2r_atom, tCrC[(None, atom_m), tile_m, tile_k], tCrC_reshape[((None, tile_k % 2, tile_k // 2), 0), atom_m, tile_m])
    r2g_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=8*cutlass.Float16.width)
    cute.copy(r2g_atom, tCrC_reshape, tCgC_reshape, pred=tCpC)
    return


@cute.jit
def static_entry(
    activations: cute.Tensor,
    filter: cute.Tensor,
    out: cute.Tensor,
    stride: cutlass.Constexpr,
    pad: cutlass.Constexpr
):
    batch_size, height, width, in_channels = activations.shape
    out_channels, filter_height, filter_width, _ = filter.shape
    _, out_height, out_width, _ = out.shape

    # ------ Tiler Config ------
    TILE_P = 8
    TILE_Q = 16
    tiler_m = (1, TILE_P, TILE_Q)
    tiler_n = 32
    tiler_k = (1, 1, 32)
    num_stages = 3

    MMA_TILE = (4, 1, 1)
    num_threads = cute.size(MMA_TILE) * 32
    # --------------------------

    v_act_lt = cute.make_layout(
        shape=(
            (batch_size, out_height, out_width),
            (filter_height, filter_width, in_channels)
        ),
        stride=(
            (height * width * in_channels, stride * width * in_channels, stride * in_channels),
            (width * in_channels, in_channels, 1)
        )
    )
    v_act = cute.domain_offset(
        ((0, 0, 0), (-pad, -pad, 0)),  # offset (r,s) to shift base pointer by -pad in h,w
        cute.make_tensor(iterator=activations.iterator.align(cute.size(tiler_k) * 2), layout=v_act_lt)
    )
    filter = cute.make_tensor(
        iterator=filter.iterator.align(cute.size(tiler_k) * 2), layout=filter.layout
    )
    grouped_filter = cute.group_modes(filter, 1, 4)
    grouped_out = cute.group_modes(out, 0, 3)
    
    sA_layout = cute.make_composed_layout(
        inner=cute.make_swizzle(b=2, m=3, s=3),
        offset=0,
        outer=cute.make_ordered_layout(
            (cute.size(tiler_m), cute.size(tiler_k), num_stages), order=(1, 0, 2)
        )
    )
    sB_layout = cute.make_composed_layout(
        inner=cute.make_swizzle(b=2, m=3, s=5),
        offset=0,
        outer=cute.make_ordered_layout(
            (cute.size(tiler_n), cute.size(tiler_k), num_stages), order=(1, 0, 2)
        )
    )

    # max 128-bit load w/ fp16 = 8x fp16
    TV_SIZE = 128 // 16
    THR_K = cute.size(tiler_k) // TV_SIZE

    tA = cute.make_layout(
        shape=((1, TILE_Q, num_threads // THR_K // TILE_Q), (1, 1, THR_K)),
        stride=((0, THR_K, TILE_Q * THR_K), (0, 0, 1))
    )
    vA = cute.make_layout(((1, 1, 1), (1, 1, TV_SIZE)), stride=((0, 0, 0), (0, 0, 1)))
    tB = cute.make_layout(
        shape=(num_threads // THR_K, (1, 1, THR_K)),
        stride=(THR_K, (0, 0, 1))
    )
    vB = cute.make_layout((1, (1, 1, TV_SIZE)), stride=(0, (0, 0, 1)))
    g2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16,
        num_bits_per_copy=TV_SIZE * cutlass.Float16.width
    )
    g2s_copy_A = cute.make_tiled_copy_tv(g2s_copy_atom, tA, vA)
    g2s_copy_B = cute.make_tiled_copy_tv(g2s_copy_atom, tB, vB)

    s2r_copy_atom_A = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4
        ),
        copy_internal_type=cutlass.Float16
    )
    s2r_copy_atom_B = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False,
            num_matrices=2
        ),
        copy_internal_type=cutlass.Float16
    )
    mma_atom = cute.make_mma_atom(
        op=cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float16,
            shape_mnk=(16, 8, 16)
        )
    )
    tiled_mma = cute.make_tiled_mma(
        op_or_atom=mma_atom,
        atom_layout_mnk=MMA_TILE,
        permutation_mnk=(
            64,
            cute.make_layout((2, 4, 4), stride=(1, 8, 2)),
            16
        )
    )

    grid_dim_m = batch_size * cute.ceil_div(out_height, TILE_P) * cute.ceil_div(out_width, TILE_Q)
    grid_dim_n = cute.ceil_div(out_channels, cute.size(tiler_n))
    grid_dim = (grid_dim_n, grid_dim_m)
    implicit_conv2d(
        v_act,
        grouped_filter,
        grouped_out,
        (height, width, stride, pad),
        sA_layout, sB_layout,
        (tiler_m, tiler_n, tiler_k),
        g2s_copy_A, g2s_copy_B,
        s2r_copy_atom_A,
        s2r_copy_atom_B,
        tiled_mma,
    ).launch(
        grid=grid_dim,
        block=(num_threads, 1, 1)
    )


@cute.jit
def dynamic_entry(
    activations_ptr: cute.Pointer,
    filter_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    N: cutlass.Int32,
    H: cutlass.Int32,
    W: cutlass.Int32,
    C: cutlass.Int32,
    K: cutlass.Int32,
    R: cutlass.Int32,
    S: cutlass.Int32,
    P: cutlass.Int32,
    Q: cutlass.Int32,
    stride: cutlass.Int32,
    pad: cutlass.Int32
):
    C = cute.assume(C, divby=8)
    activations = cute.make_tensor(activations_ptr, cute.make_ordered_layout((N, H, W, C), order=(3, 2, 1, 0)))
    filter = cute.make_tensor(filter_ptr, cute.make_ordered_layout((K, R, S, C), order=(3, 2, 1, 0)))
    out = cute.make_tensor(out_ptr, cute.make_ordered_layout((N, P, Q, K), order=(3, 2, 1, 0)))
    static_entry(activations, filter, out, stride, pad)
