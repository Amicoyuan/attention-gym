## Attention-Gym: Triton-Based Sparse and Quantization Attention

Attention-Gym is a flexible and efficient framework built on Triton, designed to help researchers and developers rapidly implement, test, and validate innovative attention mechanisms. With support for sparse and quantized attention, it provides a powerful base environment for experimenting with new algorithms and optimizing existing ones.

## Requirements

* `python>=3.9` , `torch>=2.3.0` , `triton>=3.0.0`, `NVIDIA GPUs (Compute Capability 8.0+)`
* `Notice: FP8 dtype is only supported on NVIDIA GPUs (Compute Capability 9.0+)`

## Installation

```
pip install -e.
```

## Kernels

Now Support:

* [flash_attention2](https://arxiv.org/abs/2307.08691)
* [sliding_tile_attention](https://arxiv.org/abs/2502.04507)
* [sageattn_qk_int8_pv_fp16](https://arxiv.org/abs/2410.02367)
* [sageattn_qk_int8_pv_fp8](https://arxiv.org/abs/2411.10958)
* [sparge_sageattn_qk_int8_pv_fp16](https://arxiv.org/abs/2502.18137)
* [sparge_sageattn_qk_int8_pv_fp8](https://arxiv.org/abs/2502.18137)

## How to Use

To easy use:

```
import attention_gym
out = attention_gym.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)
```

* `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` using default `tensor_layout="HND"`. For shape `(batch_size, seq_len, head_num, head_dim)`, set `tensor_layout="NHD"`.
* `is_causal` determines the use of a causal mask.

## Kernel Tests

To run the tests:

```
pytest tests/test_sageattn_qk_int8_pv_fp16.py
```

## Kernel Benchmark

To run the benchmarks:

```
python benchmarks/benchmark_sage1.py
```

## End-to-end Performance And Accuracy

Here we compare the end-to-end performance and accuracy of the original algorithm author's CUDA implementation and the attention-gym triton implementation of each algorithm.

<table>
  <tr>
    <th>Algorithm</th>
    <th>CUDA</th>
    <th>CUDA Time</th>
    <th>Triton</th>
    <th>Triton Time</th>
    <th>Env</th>
  </tr>
  <tr>
    <td>STA</td>
    <td><img src="assets/sta_cuda_2_H20.gif" width="200" alt="STA_CUDA_2_H20"></td>
    <td>1639.61s</td>
    <td><img src="assets/sta_triton_2_H20.gif" width="200" alt="STA_triton_2_H20"></td>
    <td>1853.24s</td>
    <td>wanx2.1-14B H20 2-gpus</td>
  </tr>
 <tr>
    <td>sparge_sage2</td>
    <td><img src="assets/sparge_sage2_cuda_1_H20.gif" width="200" height="300" alt="sparge_sage2_1_H20"></td>
    <td>260s</td>
    <td><img src="assets/sparge_sage2_triton_1_H20.gif" width="200" height="300" alt="sparge_sage2_triton_1_H20"></td>
    <td>268s</td>
    <td>wanx2.1-1.3B H20 1-gpu</td>
 </tr>
 <tr>
    <td>sage2</td>
    <td><img src="assets/sage2_cuda_1_H20.gif" width="200" alt="sage2_cuda_1_H20"></td>
    <td>348.95s</td>
    <td><img src="assets/sage2_triton_1_H20.gif" width="200" alt="sage2_triton_1_H20"></td>
    <td>359.94s</td>
    <td>wanx2.1-1.3B H20 1-gpu</td>
 </tr>
</table>

## Acknowledgement

We learned the design and resued some code from the following projects: [triton](https://github.com/triton-lang/triton), [FastVideo](https://github.com/hao-ai-lab/FastVideo), [SpargeAttn](https://github.com/thu-ml/SpargeAttn), [SageAttention](https://github.com/thu-ml/SageAttention)

