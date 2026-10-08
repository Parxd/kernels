from math import log2
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

B = 16
H_Q = 64  # query heads
H_KV = 8  # kv heads
GROUPS = H_Q // H_KV
S = 1024  # seq. len.
H_D = 128  # head dim.
M_D = 8192  # model dim.
BLOCK_Q, BLOCK_K = 128, 64  # query, key tiles
dtype, cute_dtype = torch.bfloat16, cute.BFloat16
torch.manual_seed(0)


@cute.kernel
def kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    sQ_layout: cute.ComposedLayout,
    sKV_layout: cute.ComposedLayout,
    tiled_copy_Q: cute.TiledCopy,
    tiled_copy_KV: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    cta_tiler = (BLOCK_Q, BLOCK_K, H_D)
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    b_coord = (bidz, None, None)  # CTAs own all kv-tiles

    gQ = cute.local_tile(
        mQ[bidx, None, bidy, None],
        tiler=cta_tiler,
        coord=b_coord,
        proj=(1, None, 1),
    )
    gK = cute.local_tile(
        mK[bidx, None, bidy // GROUPS, None],
        tiler=cta_tiler,
        coord=b_coord,
        proj=(None, 1, 1),
    )
    gV = cute.local_tile(
        mV[bidx, None, bidy // GROUPS, None],
        tiler=cta_tiler,
        coord=b_coord,
        proj=(None, 1, 1),
    )
    print(f"\tgQ: {gQ}")
    print(f"\tgK: {gK}")
    print(f"\tgV: {gV}\n")

    @cute.struct
    class SharedStorageQKV:
        q: cute.struct.Align[
            cute.struct.MemRange[cute_dtype, cute.cosize(sQ_layout)], 16
        ]
        k: cute.struct.Align[
            cute.struct.MemRange[cute_dtype, cute.cosize(sKV_layout)], 16
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[cute_dtype, cute.cosize(sKV_layout)], 16
        ]

    smem_alloc = cutlass.utils.SmemAllocator()
    smem = smem_alloc.allocate(SharedStorageQKV.size_in_bytes(), byte_alignment=16)
    sQ = SharedStorageQKV(smem).q.get_tensor(sQ_layout)
    sK = SharedStorageQKV(smem).k.get_tensor(sKV_layout)
    sV = SharedStorageQKV(smem).v.get_tensor(sKV_layout)
    print(f"\tsQ: {sQ}")
    print(f"\tsK: {sK}")
    print(f"\tsV: {sV}\n")

    # g2s copy slices
    tQ = tiled_copy_Q.get_slice(tidx)
    tK = tV = tiled_copy_KV.get_slice(tidx)
    tQgQ = tQ.partition_S(gQ)
    tKgK = tK.partition_S(gK)
    tVgV = tV.partition_S(gV)
    tQsQ = tQ.partition_D(sQ)
    tKsK = tK.partition_D(sK)
    tVsV = tV.partition_D(sV)
    # MMA slices (P := QK^T)
    thr_mma = tiled_mma.get_slice(tidx)
    tPsQ = thr_mma.partition_A(sQ)
    tPsK = thr_mma.partition_B(sK)
    tPsP = thr_mma.partition_shape_C(
        (BLOCK_Q, BLOCK_K)
    )  # get P fragment shape w/o physical sP tensor
    tOsV = thr_mma.partition_B(sV)
    tPrQ = thr_mma.make_fragment_A(tPsQ)
    tPrK = thr_mma.make_fragment_B(tPsK)
    tPrP = thr_mma.make_fragment_C(tPsP)
    tOrV = thr_mma.make_fragment_B(tOsV)
    # s2r copy slices
    s2r_copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cute_dtype
    )
    s2r_tiled_copy_Q = cute.make_tiled_copy_A(s2r_copy_atom, tiled_mma)
    s2r_tiled_copy_K = s2r_tiled_copy_V = cute.make_tiled_copy_B(
        s2r_copy_atom, tiled_mma
    )

    tsQ = s2r_tiled_copy_Q.get_slice(tidx)
    tsK = s2r_tiled_copy_K.get_slice(tidx)
    tsV = s2r_tiled_copy_V.get_slice(tidx)
    tPsQ_copy = tsQ.partition_S(sQ)
    tPsK_copy = tsK.partition_S(sK)
    tOsV_copy = tsV.partition_S(sV)
    tPrQ_copy = tsQ.retile(tPrQ)
    tPrK_copy = tsK.retile(tPrK)
    tOrV_copy = tsV.retile(tOrV)

    rowmax = cute.make_rmem_tensor(
        (2, cute.size(cute.get(tPrP.layout, [0, 1])), cute.size(tPrP, [1])),
        dtype=cute.Float32,
    )  # ((prev, current), accum. fragment rows, warp rows)
    denom = cute.make_rmem_tensor(
        (2, cute.size(cute.get(tPrP.layout, [0, 1])), cute.size(tPrP, [1])),
        dtype=cute.Float32,
    )  # ((prev, current), accum. fragment rows, warp rows)
    rowmax.fill(-cute.Float.inf)
    denom.fill(0.0)
    print(rowmax)
    print(denom)

    cute.copy(tiled_copy_Q, tQgQ[None, None, 0, 0], tQsQ[None, None, 0])
    kv_iters_full = bidz * (BLOCK_Q // BLOCK_K)  # BLOCK_K must divide BLOCK_Q
    kv_iters_masked = (bidz + 1) * (BLOCK_Q // BLOCK_K)
    for j in range(kv_iters_full):
        tPrP.fill(0.0)
        cute.copy(tiled_copy_KV, tKgK[None, None, 0, j, 0], tKsK[None, None, 0])
        cute.copy(tiled_copy_KV, tVgV[None, None, 0, j, 0], tVsV[None, None, 0])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        for k_block in range(H_D // 16):  # tensor core inst. uses K=16
            cute.copy(
                s2r_tiled_copy_Q,
                tPsQ_copy[(None, (None, k_block)), 0, 0],
                tPrQ_copy[(None, (None, k_block)), 0, 0],
            )
            cute.copy(
                s2r_tiled_copy_K,
                tPsK_copy[(None, (None, k_block)), 0, 0],
                tPrK_copy[(None, (None, k_block)), 0, 0],
            )
            cute.gemm(
                tiled_mma,
                tPrP,
                tPrQ[None, None, k_block],
                tPrK[None, None, k_block],
                tPrP,
            )


@cute.jit
def call(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor):
    q_tile = (BLOCK_Q, H_D)
    k_tile = v_tile = (BLOCK_K, H_D)
    mma_tile = (4, 1, 1)
    g2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), cute_dtype, num_bits_per_copy=128
    )
    g2s_tiled_copy = cute.make_tiled_copy_tv(
        g2s_copy_atom,
        cute.make_ordered_layout((8, 16), (1, 0)),
        cute.make_ordered_layout((1, 8), (1, 0)),
    )
    # smem major-dim = 64x bf16 = 128B = all 32x 4B banks
    swizzle_b = int(
        log2(H_D * cute_dtype.width // 128)
    )  # 128 contiguous bits in g2s copy and LdMatrix inst. per lane
    swizzle = cute.make_swizzle(b=swizzle_b, m=3, s=3)
    sQ = cute.make_composed_layout(
        swizzle, 0, outer=cute.make_ordered_layout((BLOCK_Q, H_D), order=(1, 0))
    )
    sKV = cute.make_composed_layout(
        swizzle, 0, outer=cute.make_ordered_layout((BLOCK_K, H_D), order=(1, 0))
    )
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cute_dtype, cute.Float32, (16, 8, 16)
    )  # QK^T accumulated to fp32, then cast back to bf16 for PV
    tiled_mma = cute.make_tiled_mma(
        mma_op, mma_tile, permutation_mnk=(BLOCK_Q, BLOCK_K, H_D)
    )  # TODO: add permutation on N-mode?
    grid_dims = (B, H_Q, cute.ceil_div(S, BLOCK_Q))
    block_dims = (cute.size(mma_tile) * 32, 1, 1)
    kernel(q, k, v, o, sQ, sKV, g2s_tiled_copy, g2s_tiled_copy, tiled_mma).launch(
        grid=grid_dims, block=block_dims
    )


def main():
    q = torch.randn(B, S, H_Q, H_D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, H_D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, H_D, dtype=dtype, device="cuda")
    o = torch.empty(B, S, H_Q, H_D, dtype=dtype, device="cuda")
    call(
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(o, assumed_align=16),
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
