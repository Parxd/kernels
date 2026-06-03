## kernels

a continuously expanding collection of high-performance ML-centric (so far) CUDA kernels

some of these kernels have their naive + optimized versions, with everything in-between, and some require CUTLASS CuTe

ops. supported:
- [conv2d](src/conv)
- [binary elementwise](src/elementwise)
- [fp32fp32 GEMM](src/sgemm)
- [2d transpose](src/transpose)
