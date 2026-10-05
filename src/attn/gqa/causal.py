from math import log2
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

B = 16
H_Q = 64  # query heads
H_KV = 8  # kv heads
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
    sQ: cute.ComposedLayout,
    sKV: cute.ComposedLayout,
    tiled_copy_Q: cute.TiledCopy,
    tiled_copy_KV: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()


@cute.jit
def call(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor):
    q_tile = (BLOCK_Q, H_D)
    k_tile = (BLOCK_K, H_D)
    v_tile = k_tile
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
        mma_op, mma_tile, permutation_mnk=(BLOCK_Q // cute.size(mma_tile), BLOCK_K, H_D)
    )  # TODO: add permutation on N-mode?
    grid_dims = (B, H_Q, cute.ceil_div(S, BLOCK_Q))
    block_dims = (cute.size(mma_tile) * 32, 1, 1)
    kernel(q, k, v, o, sQ, sKV, g2s_tiled_copy, g2s_tiled_copy, tiled_mma).launch(
        grid=grid_dims, block=block_dims
    )


def main():
    batch = torch.rand(B, S, M_D, dtype=dtype).to("cuda")
    q_proj = torch.rand(M_D, H_Q * H_D, dtype=dtype).to("cuda")
    k_proj = torch.rand(M_D, H_KV * H_D, dtype=dtype).to("cuda")
    v_proj = torch.rand(M_D, H_KV * H_D, dtype=dtype).to("cuda")
    q = (batch @ q_proj).reshape(B, S, H_Q, H_D)
    k = (batch @ k_proj).reshape(B, S, H_KV, H_D)
    v = (batch @ v_proj).reshape(B, S, H_KV, H_D)
    o = torch.empty(B, S, H_Q, H_D)
    call(from_dlpack(q), from_dlpack(k), from_dlpack(v), from_dlpack(o))


if __name__ == "__main__":
    main()
