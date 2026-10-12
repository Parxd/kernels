## kernels

a continuously expanding collection of high-performance ML-centric (so far) CUDA kernels for learning purposes

some of these kernels have their naive + optimized versions, with everything in-between, and some require CUTLASS CuTe (C++ and DSL versions)

ops. supported:
- [casual GQA](./attn/gqa)
- [conv2d](./conv/)
- [binary elementwise](./elementwise)
- [fp32fp32 GEMM](./sgemm)
- [2d transpose](./transpose)
