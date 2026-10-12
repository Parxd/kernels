# Elementwise Binary Kernels

| **Kernel ID** |  **Type** | **Technique**            | **Constraints**                        |
|---------------|-----------|--------------------------|---------------------------------------------|
| 0             | CUTLASS   | (1:1) standard thread-value loads | Dims. must be divisible by block shape      |
| 1             | CUTLASS   | (1:4) vectorized thread-value loads | Dims. must be divisible by block shape      |
| 2             | CUTLASS   | (1:4) vectorized + predicated thread-value loads| Dims. must be divisible by vector load size |