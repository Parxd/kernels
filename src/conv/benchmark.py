import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from c32k32 import static_entry, dynamic_entry

WARMUP_ITERS = 10
TIMED_ITERS = 50


def time_kernel_avg(fn, *args):
    for _ in range(WARMUP_ITERS):
        fn(*args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(TIMED_ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED_ITERS


def main():
    STRIDE, PAD = 1, 0
    N, H, W, C = 4, 80, 80, 64
    K, R, S = 128, 3, 3
    P = (H + 2 * PAD - R) // STRIDE + 1
    Q = (W + 2 * PAD - S) // STRIDE + 1
    activations = torch.randn((N, H, W, C), dtype=torch.float16).cuda()
    weight = torch.randn((K, R, S, C), dtype=torch.float16).cuda()
    out = torch.empty((N, P, Q, K), dtype=torch.float16).cuda()
    act_cute = from_dlpack(activations)
    weight_cute = from_dlpack(weight)
    out_cute = from_dlpack(out)
    compiled = cute.compile(static_entry, act_cute, weight_cute, out_cute, STRIDE, PAD)
    act_nchw = activations.permute(0, 3, 1, 2).contiguous()
    weight_nchw = weight.permute(0, 3, 1, 2).contiguous()
    def run_ref():
        return torch.nn.functional.conv2d(
            act_nchw, weight_nchw,
            stride=STRIDE,
            padding=PAD
        )
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    with torch.backends.cudnn.flags(enabled=True, benchmark=True):
        for _ in range(WARMUP_ITERS):
            run_ref()
        torch.cuda.synchronize()
        kernel_avg_ms = time_kernel_avg(compiled, act_cute, weight_cute, out_cute)
        torch.cuda.synchronize()
        ref_avg_ms = time_kernel_avg(run_ref)
    ref = run_ref().permute(0, 2, 3, 1).contiguous()
    print(f"kernel:         {kernel_avg_ms:.4f} ms")
    print(f"cudnn:          {ref_avg_ms:.4f} ms")
    print(f"speedup:        {ref_avg_ms / kernel_avg_ms:.3f}x")
    print(f"max abs. err.:  {(out - ref).abs().max().item():.6f}")

    # dynamic entry, compile once for all shapes at the cost of performance
    # STRIDE, PAD = 2, 1
    # N, H, W, C = 1, 320, 320, 32
    # K, R, S = 64, 3, 3
    # P = (H + 2 * PAD - R) // STRIDE + 1
    # Q = (W + 2 * PAD - S) // STRIDE + 1
    # activations = torch.randn((N, H, W, C), dtype=torch.float16).cuda()
    # weight = torch.randn((K, R, S, C), dtype=torch.float16).cuda()
    # out = torch.empty((N, P, Q, K), dtype=torch.float16).cuda()
    # compiled = cute.compile(
    #     dynamic_entry,
    #     make_ptr(cutlass.Float16, activations.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #     make_ptr(cutlass.Float16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #     make_ptr(cutlass.Float16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #     N, H, W, C, K, R, S, P, Q, STRIDE, PAD
    # )
    # act_nchw = activations.permute(0, 3, 1, 2).contiguous()
    # weight_nchw = weight.permute(0, 3, 1, 2).contiguous()
    # def run_ref():
    #     return torch.nn.functional.conv2d(
    #         act_nchw, weight_nchw,
    #         stride=STRIDE,
    #         padding=PAD
    #     )
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # with torch.backends.cudnn.flags(enabled=True, benchmark=True):
    #     for _ in range(WARMUP_ITERS):
    #         run_ref()
    #     torch.cuda.synchronize()
    #     kernel_avg_ms = time_kernel_avg(
    #         compiled,
    #         make_ptr(cutlass.Float16, activations.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #         make_ptr(cutlass.Float16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #         make_ptr(cutlass.Float16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #         N, H, W, C, K, R, S, P, Q, STRIDE, PAD
    #     )
    #     torch.cuda.synchronize()
    #     ref_avg_ms = time_kernel_avg(run_ref)
    # ref = run_ref().permute(0, 2, 3, 1).contiguous()
    # print(f"kernel:         {kernel_avg_ms:.4f} ms")
    # print(f"cudnn:          {ref_avg_ms:.4f} ms")
    # print(f"speedup:        {ref_avg_ms / kernel_avg_ms:.3f}x")
    # print(f"max abs. err.:  {(out - ref).abs().max().item():.6f}")


if __name__ == "__main__":
    main()
