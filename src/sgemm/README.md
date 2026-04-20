# FP32 General Matrix-Multiply Kernels

| **Kernel ID**                              |  **Type** | **Notes**                                                             |
|--------------------------------------------|-----------|-----------------------------------------------------------------------|
| 0 - [naive.cu](./cuda/naive.cu) | CUDA      | Naive implementation                                                  |
| 1 - [smem.cu](./cuda/smem.cu)                            | CUDA      | SMEM tiling                                                           |
| 2 - [onedim_tile.cu](./cuda/onedim_tile.cu)                     | CUDA      | SMEM tiling + 1-dim. thread tile MMA                                  |
| 3 - [twodim_tile.cu](./cuda/twodim_tile.cu)                     | CUDA      | SMEM tiling + 2-dim. thread tile MMA                                  |
| 4 - [vectorize.cu](./cuda/vectorize.cu)                       | CUDA      | SMEM tiling + 2-dim. thread tile MMA + vectorized memory transactions |
| 5 - [128x128x8_cg.cu](./cuda/128x128x8_cg.cu)                    | CUDA      | SMEM tiling + 2-dim. thread tile MMA + vectorized memory transactions + "warptiling" w/ [CG API](https://developer.nvidia.com/blog/cooperative-groups/) |
| 6 - [128x128x16.cu](./cuda/128x128x16.cu)                      | CUDA      | SMEM tiling + 2-dim. thread tile MMA + vectorized memory transactions + "warptiling" |
| 7 - [siboehm.cu](./cuda/siboehm.cu)                         | CUDA      | Simon Boehm's [kernel](https://siboehm.com/articles/22/CUDA-MMM)      |
| 8                                          | cuBLAS    | NVIDIA's proprietary closed-source BLAS implementation                |

Below are performance metrics for kernels 4 and above (kernels 0 - 3 are for learning purposes). Each benchmark was run on an NVIDIA RTX3070 Mobile averaged over 50 trials, with the fastest for each matrix size bolded:

<table>
  <tr style="background-color:#DDDDDD;">
    <th>Kernel ID</th><th>Matrix Size (M=N=K)</th><th>GFLOPS</th>
  </tr>

  <!-- 512 -->
  <tr style="background-color:#E6F2FF;"><td>4</td><td>512</td><td>2836.44</td></tr>
  <tr style="background-color:#E6F2FF;"><td>5</td><td>512</td><td>1681.49</td></tr>
  <tr style="background-color:#E6F2FF;"><td>6</td><td>512</td><td>2376.65</td></tr>
  <tr style="background-color:#E6F2FF;"><td>7</td><td>512</td><td>1820.19</td></tr>
  <tr style="background-color:#E6F2FF;"><td><b>8</b></td><td><b>512</b></td><td><b>4568.56</b></td></tr>

  <!-- 1024 -->
  <tr style="background-color:#E9FFE6;"><td>4</td><td>1024</td><td>4097.92</td></tr>
  <tr style="background-color:#E9FFE6;"><td>5</td><td>1024</td><td>4406.15</td></tr>
  <tr style="background-color:#E9FFE6;"><td>6</td><td>1024</td><td>5901.82</td></tr>
  <tr style="background-color:#E9FFE6;"><td>7</td><td>1024</td><td>4547.16</td></tr>
  <tr style="background-color:#E9FFE6;"><td><b>8</b></td><td><b>1024</b></td><td><b>7293.93</b></td></tr>

  <!-- 2048 -->
  <tr style="background-color:#FFF7CC;"><td>4</td><td>2048</td><td>5030.32</td></tr>
  <tr style="background-color:#FFF7CC;"><td>5</td><td>2048</td><td>5453.03</td></tr>
  <tr style="background-color:#FFF7CC;"><td><b>6</b></td><td><b>2048</b></td><td><b>7745.14</b></td></tr>
  <tr style="background-color:#FFF7CC;"><td>7</td><td>2048</td><td>5426.43</td></tr>
  <tr style="background-color:#FFF7CC;"><td>8</td><td>2048</td><td>7171.22</td></tr>

  <!-- 4096 -->
  <tr style="background-color:#F2E6FF;"><td>4</td><td>4096</td><td>4687.46</td></tr>
  <tr style="background-color:#F2E6FF;"><td>5</td><td>4096</td><td>5971.47</td></tr>
  <tr style="background-color:#F2E6FF;"><td><b>6</b></td><td><b>4096</b></td><td><b>7481.53</b></td></tr>
  <tr style="background-color:#F2E6FF;"><td>7</td><td>4096</td><td>5984.16</td></tr>
  <tr style="background-color:#F2E6FF;"><td>8</td><td>4096</td><td>6476.98</td></tr>
</table>

### miscellaneous details about kernels 5+ ...
- Kernels 5 & 6 do **NOT** work for arbitrary matrices; M, N, K must be multiples of 128. PLEASE do not use in production
- Kernel 5
    - Uses Cooperative Groups API for more readable/modular kernel code
    - Uses SMEM tiling size of (BM, BN, BK) = (128, 128, 8)
    - Uses "warptiling" size of (WM, WN) = (64, 64)
    - Uses thread tile MMA size of (TM x WIM, TN x WIN) = (4 x 2, 4 x 4) = (8, 16)
    - Launches 128 threads per CTA
- Kernel 6
    - Uses SMEM tiling size of (BM, BN, BK) = (128, 128, 16)
    - Uses "warptiling" size of (WM, WN) = (64, 64)
    - Uses thread tile MMA size of (TM x WIM, TN x WIN) = (4 x 2, 4 x 8) = (8, 32)
    - Launches 128 threads per CTA
    - Uses minimal inline PTX to guarantee vectorized loads from SMEM → Registerfile
