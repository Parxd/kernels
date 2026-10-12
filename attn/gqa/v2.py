from math import log2, sqrt
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
BLOCK_Q, BLOCK_K = 64, 64  # query, key tiles
LOG2_E = 1.44269504089
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
    # CTAs own all kv-tiles
    b_coord = (bidz, None, 0)  # 0 in last mode since H_D == cta_tiler[2]

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
    gO = cute.local_tile(
        mO[bidx, None, bidy, None],
        tiler=cta_tiler,
        coord=b_coord,
        proj=(1, None, 1),
    )

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

    smem_alloc = cutlass.memory.SmemAllocator()
    smem = smem_alloc.allocate(SharedStorageQKV.size_in_bytes(), byte_alignment=16)
    sQ = SharedStorageQKV(smem).q.get_tensor(sQ_layout)
    sK = SharedStorageQKV(smem).k.get_tensor(sKV_layout)
    sV = SharedStorageQKV(smem).v.get_tensor(sKV_layout)
    sVt = cute.composition(sV, cute.make_layout((H_D, BLOCK_K), stride=(BLOCK_K, 1)))

    print(f"\tgQ: {gQ}")
    print(f"\tgK: {gK}")
    print(f"\tgV: {gV}")
    print(f"\tgO: {gO}\n")
    print(f"\tsQ: {sQ}")
    print(f"\tsK: {sK}")
    print(f"\tsV: {sV}")
    print(f"\tsVt: {sVt}\n")

    # g2s copy slices
    tQ = tiled_copy_Q.get_slice(tidx)
    tK = tV = tiled_copy_KV.get_slice(tidx)
    tQgQ = tQ.partition_S(gQ)
    tKgK = tK.partition_S(gK)
    tVgV = tV.partition_S(gV)
    tQsQ = tQ.partition_D(sQ)
    tKsK = tK.partition_D(sK)
    tVsV = tV.partition_D(sV)
    # MMA slices (P := QK^T, O := PV)
    thr_mma = tiled_mma.get_slice(tidx)
    tPsQ = thr_mma.partition_A(sQ)
    tPsK = thr_mma.partition_B(sK)
    tPsP = thr_mma.partition_shape_C((BLOCK_Q, BLOCK_K))
    cP = cute.make_identity_tensor((BLOCK_Q, BLOCK_K))
    tPcP = thr_mma.partition_C(cP)
    tOgO = thr_mma.partition_C(gO)
    tOsV = thr_mma.partition_B(sVt)
    tPrQ = thr_mma.make_fragment_A(tPsQ)
    tPrK = thr_mma.make_fragment_B(tPsK)
    tPrP = thr_mma.make_fragment_C(tPsP)
    tOrV = thr_mma.make_fragment_B(tOsV)
    tOrO = thr_mma.make_fragment_C(tOgO)
    tOrO.fill(0.0)
    # s2r copy slices
    s2r_copy_atom_QK = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cute_dtype
    )
    s2r_copy_atom_V = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cute_dtype
    )
    s2r_tiled_copy_Q = cute.make_tiled_copy_A(s2r_copy_atom_QK, tiled_mma)
    s2r_tiled_copy_K = cute.make_tiled_copy_B(s2r_copy_atom_QK, tiled_mma)
    s2r_tiled_copy_V = cute.make_tiled_copy_B(s2r_copy_atom_V, tiled_mma)

    tsQ = s2r_tiled_copy_Q.get_slice(tidx)
    tsK = s2r_tiled_copy_K.get_slice(tidx)
    tsV = s2r_tiled_copy_V.get_slice(tidx)
    tPsQ_copy = tsQ.partition_S(sQ)
    tPsK_copy = tsK.partition_S(sK)
    tOsV_copy = tsV.partition_S(sVt)
    tPrQ_copy = tsQ.retile(tPrQ)
    tPrK_copy = tsK.retile(tPrK)
    tOrV_copy = tsV.retile(tOrV)

    row_max = cute.make_rmem_tensor(
        (cute.size(tPrP, mode=[0, 1]), cute.size(tPrP, mode=[1])),
        dtype=cute.Float32,
    )  # current KV-tile row maxes
    row_max_prev = cute.make_rmem_tensor(
        (cute.size(tPrP, mode=[0, 1]), cute.size(tPrP, mode=[1])),
        dtype=cute.Float32,
    )  # previous KV-tile row maxes
    denom = cute.make_rmem_tensor(
        (cute.size(tPrP, mode=[0, 1]), cute.size(tPrP, mode=[1])),
        dtype=cute.Float32,
    )
    log2_softmax_scale = cute.Float32(LOG2_E * cute.math.rsqrt(cute.Float32(H_D)))
    row_max.fill(-cute.Float.inf)
    denom.fill(0.0)

    cute.copy(tiled_copy_Q, tQgQ[None, None, 0], tQsQ[None, None, 0])
    kv_iters_full = bidz * (BLOCK_Q // BLOCK_K)  # BLOCK_K must divide BLOCK_Q
    kv_iters_masked = (bidz + 1) * (BLOCK_Q // BLOCK_K)
    for j in range(kv_iters_masked):
        tPrP.fill(0.0)
        cute.copy(tiled_copy_KV, tKgK[None, None, 0, j], tKsK[None, None, 0])
        cute.copy(tiled_copy_KV, tVgV[None, None, 0, j], tVsV[None, None, 0])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        for k_block in cutlass.range_constexpr(H_D // 16):
            # TODO: could move this outside KV-loop
            cute.copy(
                s2r_tiled_copy_Q,
                tPsQ_copy[None, None, k_block],
                tPrQ_copy[None, None, k_block],
            )
            cute.copy(
                s2r_tiled_copy_K,
                tPsK_copy[None, None, k_block],
                tPrK_copy[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tPrP,
                tPrQ[None, None, k_block],
                tPrK[None, None, k_block],
                tPrP,
            )
        if j >= kv_iters_full:
            for i in cutlass.range_constexpr(cute.size(tPrP)):
                x, y = tPcP[i][0], tPcP[i][1]
                if y + (j * BLOCK_K) > x + (bidz * BLOCK_Q):
                    tPrP[i] = -cute.Float32.inf
        row_max_prev.store(row_max.load())
        for mma_m in cutlass.range_constexpr(cute.size(row_max, mode=[1])):  # MMA_M
            for frg_m in cutlass.range_constexpr(cute.size(row_max, mode=[0])):  # Frg_M
                thr_row = tPrP[(None, frg_m), mma_m, None]
                thr_max = thr_row.load().reduce(
                    cute.ReductionOp.MAX,
                    row_max_prev[frg_m, mma_m],
                    reduction_profile=0,
                )
                row_max[frg_m, mma_m] = cute.arch.warp_reduction(
                    thr_max, cute.math.max, threads_in_group=4
                )
                thr_row.store(
                    cute.math.exp2(
                        thr_row.load() * log2_softmax_scale
                        - row_max[frg_m, mma_m] * log2_softmax_scale
                    )
                )
                denom_norm = cute.math.exp2(
                    (row_max_prev[frg_m, mma_m] - row_max[frg_m, mma_m])
                    * log2_softmax_scale
                )
                thr_sum = thr_row.load().reduce(
                    cute.ReductionOp.ADD,
                    0.0,
                    reduction_profile=0,
                )
                denom[frg_m, mma_m] = (
                    denom_norm * denom[frg_m, mma_m] + thr_sum
                )  # defer warp-level sum

                o_row = tOrO[(None, frg_m), mma_m, None]
                o_row.store(o_row.load() * denom_norm)
        tPrP_bf16 = cute.make_rmem_tensor_like(tPrP, dtype=cute.BFloat16)
        tPrP_bf16.store(tPrP.load().to(cute.BFloat16))
        tPrP_lt = cute.logical_divide(tPrP.layout, (None, None, 2))
        tOrP = cute.make_tensor(
            tPrP_bf16.iterator,
            cute.make_layout(
                shape=(
                    (tPrP_lt.shape[0], tPrP_lt.shape[2][0]),
                    tPrP_lt.shape[1],
                    tPrP_lt.shape[2][1],
                ),
                stride=(
                    (tPrP_lt.stride[0], tPrP_lt.stride[2][0]),
                    tPrP_lt.stride[1],
                    tPrP_lt.stride[2][1],
                ),
            ),
        )
        for k_block in cutlass.range_constexpr(BLOCK_K // 16):
            cute.copy(
                s2r_tiled_copy_V,
                tOsV_copy[None, None, k_block],
                tOrV_copy[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tOrO,
                tOrP[None, None, k_block],
                tOrV[None, None, k_block],
                tOrO,
            )
        cute.arch.sync_threads()
    for mma_m in cutlass.range_constexpr(cute.size(tOgO, mode=[1])):
        for frg_m in cutlass.range_constexpr(cute.size(tOgO, mode=[0, 1])):
            denom_final = cute.arch.warp_reduction(
                denom[frg_m, mma_m], cute.math.add, threads_in_group=4
            )
            thr_row = tOrO[(None, frg_m), mma_m, None]
            thr_row.store(thr_row.load() * cute.math.rcp(denom_final))
    tOrO_bf16 = cute.make_rmem_tensor_like(tOrO, dtype=cute.BFloat16)
    tOrO_bf16.store(tOrO.load().to(cute.BFloat16))
    r2g_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        cute.BFloat16,
        num_bits_per_copy=(2 * cute.BFloat16.width),
    )
    cute.copy(r2g_copy_atom, tOrO_bf16, tOgO)


@cute.jit
def call(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor):
    mma_inst = (16, 8, 16)
    mma_tile = (4, 1, 1)
    g2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), cute_dtype, num_bits_per_copy=128
    )
    g2s_tiled_copy = cute.make_tiled_copy_tv(
        g2s_copy_atom,
        cute.make_ordered_layout((8, 16), (1, 0)),
        cute.make_ordered_layout((1, 8), (1, 0)),
    )
    swizzle = cute.make_swizzle(b=3, m=3, s=4)
    sQ = cute.make_composed_layout(
        swizzle, 0, outer=cute.make_ordered_layout((BLOCK_Q, H_D), order=(1, 0))
    )
    sKV = cute.make_composed_layout(
        swizzle, 0, outer=cute.make_ordered_layout((BLOCK_K, H_D), order=(1, 0))
    )
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cute_dtype, cute.Float32, mma_inst
    )  # QK^T accumulated to fp32, then cast back to bf16 for PV
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        mma_tile,
        permutation_mnk=(
            mma_inst[0] * cute.size(mma_tile),
            mma_inst[1] * 2,
            mma_inst[2],
        ),
    )
    grid_dims = (B, H_Q, cute.ceil_div(S, BLOCK_Q))
    block_dims = (cute.size(mma_tile) * 32, 1, 1)
    kernel(q, k, v, o, sQ, sKV, g2s_tiled_copy, g2s_tiled_copy, tiled_mma).launch(
        grid=grid_dims, block=block_dims
    )


def torch_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    kv_limit: int = 0,
) -> torch.Tensor:
    if kv_limit:
        k = k[:, 0:kv_limit, :, :]
        v = v[:, 0:kv_limit, :, :]
    q_perm = q.permute(0, 2, 1, 3)
    k_perm = k.permute(0, 2, 1, 3).repeat_interleave(GROUPS, 1)
    v_perm = v.permute(0, 2, 1, 3).repeat_interleave(GROUPS, 1)
    s = torch.matmul(q_perm, k_perm.transpose(2, 3)) / sqrt(H_D)
    if causal:
        s = s.masked_fill(
            torch.ones((S, S), dtype=torch.bool).to('cuda').triu(1),
            -torch.inf,
        )
    p = torch.softmax(s, dim=3)
    out = torch.matmul(p, v_perm)
    return out.permute(0, 2, 1, 3).to(dtype=dtype)


def main():
    q = torch.randn(B, S, H_Q, H_D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, H_D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, H_D, dtype=dtype, device="cuda")
    o = torch.empty(B, S, H_Q, H_D, dtype=dtype, device="cuda")
    compiled = cute.compile(
        call,
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        options="remarks{ptx}",
    )
    compiled(
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(o, assumed_align=16),
    )
    torch.cuda.synchronize()

    # o_torch_naive = torch_naive(q, k, v, causal=True)
    # torch.testing.assert_close(o.float(), o_torch_naive.float(), atol=2e-2, rtol=2e-2)

    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
        enable_gqa=True,
    ).permute(0, 2, 1, 3)    
    torch.testing.assert_close(o.float(), o_sdpa.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
