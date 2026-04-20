import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
compiled = cute.compile(foo, a, 3)
compiled(a)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
compiled(b)                # Second call with dynamic layout