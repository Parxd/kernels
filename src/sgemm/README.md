## fp32 GEMM kernels

the kernels in `cuda/` are written in raw CUDA and closely follow Simon Boehm's [blogpost](https://siboehm.com/articles/22/CUDA-MMM) about optimizing SGEMM

those in `cutlass/` are written with the CuTe C++ API and are significantly more optimized (introduced pipelining) and readable
