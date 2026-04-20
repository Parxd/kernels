import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def entry(batch_size: int, height: int, width: int, in_channels: int,
          filter_height: int, filter_width: int, out_channels: int,
          stride: int = 2, padding: int = 1):
    P = (height + 2 * padding - filter_height) // stride + 1
    Q = (width + 2 * padding - filter_width) // stride + 1

    A = cute.make_ordered_layout((1, 640, 640, 3), order=(3, 2, 1, 0))


def main():
    N, H, W, C = 1, 640, 640, 3
    K, R, S = 64, 3, 3

    activations = torch.rand((N, H, W, C), dtype=torch.float16, device='cuda')
    filter = torch.rand((K, R, S, C), dtype=torch.float16, device='cuda')
    ref = torch.conv2d(activations, filter)

    entry(1, 640, 640, 3, 3, 3, 64)


if __name__ == "__main__":
    main()
