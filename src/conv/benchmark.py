import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from c32k32 import static_entry, dynamic_entry
from dataclasses import dataclass

WARMUP_ITERS = 10
TIMED_ITERS = 50
M_TILER_THRESHOLD = 1600  # TODO: tune to find optimal value here


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


@dataclass
class ConvShape:
    name: str
    N: int
    H: int
    W: int
    C: int
    K: int
    R: int
    S: int
    STRIDE: int = 1
    PAD: int = 0


def common_shapes():
    shapes = []

    resnet_3x3 = [
        ("resnet_stage1", 8, 56, 56, 64, 64),
        ("resnet_stage2", 8, 28, 28, 128, 128),
        ("resnet_stage3", 8, 14, 14, 256, 256),
        ("resnet_stage4", 8, 7,  7,  512, 512),
    ]
    for name, N, H, W, C, K in resnet_3x3:
        shapes.append(ConvShape(name, N, H, W, C, K, 3, 3, STRIDE=1, PAD=1))

    downsample_3x3 = [
        ("downsample_1", 8, 112, 112, 64, 128),
        ("downsample_2", 8, 56,  56,  128, 256),
        ("downsample_3", 8, 28,  28,  256, 512),
    ]
    for name, N, H, W, C, K in downsample_3x3:
        shapes.append(ConvShape(name, N, H, W, C, K, 3, 3, STRIDE=2, PAD=1))

    pointwise = [
        ("pointwise_1", 8, 56, 56, 64, 256),
        ("pointwise_2", 8, 28, 28, 128, 512),
        ("pointwise_3", 8, 14, 14, 256, 1024),
    ]
    for name, N, H, W, C, K in pointwise:
        shapes.append(ConvShape(name, N, H, W, C, K, 1, 1, STRIDE=1, PAD=0))

    shapes.append(ConvShape("inception_5x5", 8, 28, 28, 192, 32, 5, 5, STRIDE=1, PAD=2))

    shapes.append(ConvShape("yolo_head", 8, 80, 80, 256, 256, 3, 3, 1, 1))
    shapes.append(ConvShape("yolo_neck", 8, 40, 40, 512, 256, 1, 1, 1, 0))
    shapes.append(ConvShape("yolov11_layer2", 8, 320, 320, 32, 64, 3, 3, 2, 1))
    shapes.append(ConvShape("yolov11_layer3", 8, 160, 160, 64, 64, 3, 3, 1, 0))

    shapes.append(ConvShape("vgg_early", 8, 224, 224, 64, 64, 3, 3, 1, 1))
    shapes.append(ConvShape("vgg_mid", 8, 112, 112, 128, 128, 3, 3, 1, 1))

    shapes.append(ConvShape("unet_enc", 8, 128, 128, 64, 128, 3, 3, 1, 1))
    shapes.append(ConvShape("unet_dec", 8, 128, 128, 128, 64, 3, 3, 1, 1))

    shapes.append(ConvShape("m-test1", 1, 30, 30, 128, 64, 3, 3, 1, 1))
    shapes.append(ConvShape("m-test2", 2, 40, 40, 128, 64, 3, 3, 1, 1))

    return shapes


def run_one(shape: ConvShape):
    N, H, W, C = shape.N, shape.H, shape.W, shape.C
    K, R, S = shape.K, shape.R, shape.S
    STRIDE, PAD = shape.STRIDE, shape.PAD
    P = (H + 2 * PAD - R) // STRIDE + 1
    Q = (W + 2 * PAD - S) // STRIDE + 1

    activations = torch.randn((N, H, W, C), dtype=torch.float16).cuda()
    weight = torch.randn((K, R, S, C), dtype=torch.float16).cuda()
    out = torch.empty((N, P, Q, K), dtype=torch.float16).cuda()
    act_cute = from_dlpack(activations)
    weight_cute = from_dlpack(weight)
    out_cute = from_dlpack(out)
    if shape.N * shape.H * shape.W < M_TILER_THRESHOLD:
        compiled = cute.compile(static_entry, act_cute, weight_cute, out_cute, STRIDE, PAD, 64)
    else:
        compiled = cute.compile(static_entry, act_cute, weight_cute, out_cute, STRIDE, PAD, 128)
    act_nchw = activations.permute(0, 3, 1, 2).contiguous()
    weight_nchw = weight.permute(0, 3, 1, 2).contiguous()
    def run_ref():
        return torch.nn.functional.conv2d(
            act_nchw, weight_nchw, stride=STRIDE, padding=PAD
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
    max_err = (out - ref).abs().max().item()
    return {
        "name": shape.name,
        "shape": f"N={N} H={H} W={W} C={C} K={K} R={R} S={S} stride={STRIDE} pad={PAD}",
        "kernel_ms": kernel_avg_ms,
        "cudnn_ms": ref_avg_ms,
        "speedup": ref_avg_ms / kernel_avg_ms,
        "max_err": max_err,
    }


def main():
    results = []
    for shape in common_shapes():
        print(f"----- {shape.name} -----")
        try:
            r = run_one(shape)
            results.append(r)
            print(f"  shape:       {r['shape']}")
            print(f"  kernel:      {r['kernel_ms']:.4f} ms")
            print(f"  cudnn:       {r['cudnn_ms']:.4f} ms")
            print(f"  speedup:     {r['speedup']:.3f}x")
            print(f"  max abs err: {r['max_err']:.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            torch.cuda.empty_cache()
        print()

    if results:
        print("----- Summary -----")
        header = f"{'name':<18}{'kernel(ms)':>12}{'cudnn(ms)':>12}{'speedup':>10}{'max_err':>10}"
        print(header)
        for r in results:
            print(f"{r['name']:<18}{r['kernel_ms']:>12.4f}{r['cudnn_ms']:>12.4f}"
                  f"{r['speedup']:>10.3f}{r['max_err']:>10.6f}")
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print(f"\naverage speedup: {avg_speedup:.3f}x")


if __name__ == "__main__":
    main()
