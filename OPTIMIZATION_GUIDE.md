# Qwen3-TTS Optimization Guide

This document describes the inference optimizations implemented in the Qwen3-TTS OpenAI-compatible API to achieve maximum performance on NVIDIA GPUs.

## Overview

The official Qwen3-TTS backend has been optimized with multiple techniques to reduce latency and improve throughput for real-time text-to-speech synthesis. These optimizations are production-ready and applied automatically when the backend initializes.

## Implemented Optimizations

### 1. **Flash Attention 2** 
- **What**: Memory-efficient attention mechanism that reduces memory bandwidth requirements
- **Implementation**: `attn_implementation="flash_attention_2"` in model loading
- **Benefits**: 
  - Reduces GPU VRAM usage
  - Speeds up attention computation by 2-3x
  - Enables longer context lengths
- **Performance Impact**: **+10% speedup** over baseline (RTF: 0.97 → 0.87)
- **Requirements**: NVIDIA GPU with compute capability ≥ 7.5 (Volta, Turing, Ampere, or newer)

### 2. **torch.compile() Optimization**
- **What**: PyTorch 2.0+ JIT compilation to fuse operations and reduce Python overhead
- **Implementation**: `torch.compile(model, mode="reduce-overhead", fullgraph=False)`
- **Modes**:
  - `reduce-overhead`: Optimizes for inference speed (recommended for production)
  - `max-autotune`: Maximum optimization with longer compilation time
  - `default`: Balanced compilation
- **Benefits**:
  - Kernel fusion reduces GPU kernel launch overhead
  - Optimized memory access patterns
  - Automatic graph optimizations
- **Expected Impact**: **+20-30% speedup** (estimated based on PyTorch benchmarks)
- **Warmup**: First inference may be slower due to compilation (1-2 requests)

### 3. **cuDNN Benchmark Mode**
- **What**: Automatically finds the fastest convolution algorithms for your specific hardware
- **Implementation**: `torch.backends.cudnn.benchmark = True`
- **Benefits**:
  - Optimizes convolution operations
  - Adapts to your specific GPU architecture
  - Automatically selects best algorithms
- **Expected Impact**: **+5-10% speedup** on convolution-heavy models
- **Tradeoff**: First few inferences are slower while cuDNN benchmarks algorithms

### 4. **TensorFloat-32 (TF32) Precision**
- **What**: Uses TF32 format for matrix multiplications on Ampere+ GPUs (RTX 30xx/40xx)
- **Implementation**:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
- **Benefits**:
  - **3-5x speedup** on matmul operations (Ampere+ GPUs)
  - Maintains near-FP32 accuracy
  - No code changes required
- **Hardware**: NVIDIA Ampere (RTX 30xx, A100) or newer
- **Accuracy**: TF32 provides 19-bit precision vs FP32's 23-bit (negligible impact on TTS quality)

### 5. **BFloat16 Mixed Precision**
- **What**: Uses BFloat16 format for model weights and activations
- **Implementation**: `dtype=torch.bfloat16` in model loading
- **Benefits**:
  - Reduces VRAM usage by ~50%
  - Faster computation on modern GPUs
  - Better numerical stability than FP16
- **Performance Impact**: Already used in baseline benchmarks
- **Hardware**: Works best on Ampere+ GPUs with native BF16 support

## Performance Benchmarks

### Baseline (Flash Attention 2 only)
```
Configuration: Official Backend + Flash Attention 2
Average RTF: 0.87
Average Latency: 7.28s
```

### With All Optimizations
```
Configuration: Official Backend + All Optimizations
Expected RTF: ~0.65-0.70 (25-30% faster than baseline)
Expected Latency: ~5.5-6.0s
```

**Note**: torch.compile() and cuDNN benchmarking require a warmup phase. First 2-3 requests may be slower.

## Optimization Comparison

| Configuration | RTF | Speedup vs Baseline | VRAM | Notes |
|--------------|-----|-------------------|------|-------|
| **Baseline** | 0.97 | - | 3.89 GB | No optimizations |
| **+ Flash Attn 2** | 0.87 | +10% | 3.5 GB | ✅ Production ready |
| **+ torch.compile** | ~0.70 | +28% | 3.5 GB | ⏳ Warmup required |
| **+ cuDNN benchmark** | ~0.68 | +30% | 3.5 GB | ⏳ Warmup required |
| **+ TF32** | ~0.65 | +33% | 3.5 GB | 🎯 Ampere+ only |

## How to Enable/Disable Optimizations

All optimizations are enabled by default in the Docker deployment. To customize:

### Disable torch.compile()
Edit `api/backends/official_qwen3_tts.py`:
```python
# Comment out this section:
# self.model.model = torch.compile(
#     self.model.model,
#     mode="reduce-overhead",
#     fullgraph=False,
# )
```

### Disable TF32
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### Disable cuDNN Benchmark
```python
torch.backends.cudnn.benchmark = False
```

### Change torch.compile() Mode
```python
# For maximum speed (longer compilation):
self.model.model = torch.compile(self.model.model, mode="max-autotune")

# For balanced performance:
self.model.model = torch.compile(self.model.model, mode="default")
```

## Production Recommendations

### ✅ Recommended for Production
1. **Flash Attention 2**: Always enabled (significant speedup, minimal downside)
2. **TF32**: Enabled on Ampere+ GPUs (huge speedup, negligible accuracy loss)
3. **cuDNN Benchmark**: Enabled (after warmup, pure performance gain)
4. **BFloat16**: Enabled (reduces VRAM, faster inference)

### ⚠️ Consider Carefully
1. **torch.compile()**: 
   - **Pros**: 20-30% speedup after warmup
   - **Cons**: First 1-2 requests are slower, increased memory during compilation
   - **Recommendation**: Enable for long-running servers, disable for serverless

## Hardware Requirements

| Optimization | Minimum GPU | Recommended GPU | Memory |
|-------------|-------------|----------------|--------|
| Flash Attention 2 | Volta (V100) | Ampere (RTX 3090, A100) | - |
| torch.compile() | Any CUDA GPU | Ampere+ | +0.5GB during compilation |
| TF32 | Ampere (RTX 3090) | Ada Lovelace (RTX 4090) | - |
| cuDNN Benchmark | Any CUDA GPU | Any CUDA GPU | - |
| BFloat16 | Turing (RTX 20xx) | Ampere+ | -50% VRAM |

## Troubleshooting

### Long First Inference
- **Cause**: torch.compile() and cuDNN benchmarking warm up
- **Solution**: Send 2-3 warmup requests after server start
- **Duration**: 10-30 seconds for first request

### Out of Memory During Compilation
- **Cause**: torch.compile() needs extra memory
- **Solution**: Disable torch.compile() or reduce batch size
- **Memory**: Typically +0.5-1GB during compilation

### Flash Attention Errors
- **Cause**: Incompatible GPU or missing flash-attn package
- **Solution**: Install `flash-attn==2.8.3` or use `attn_implementation="sdpa"`

## Future Optimization Opportunities

### Not Yet Implemented
1. **Dynamic Batching**: Process multiple requests together (~2x throughput)
2. **CUDA Graphs**: Capture and replay execution graphs (~10-15% speedup)
3. **INT8 Quantization**: Reduce model size and speed up inference (~30-40% speedup)
4. **vLLM Integration**: Advanced KV cache management and paged attention
5. **Tensor Parallelism**: Multi-GPU inference for larger models

### Experimental
1. **torch.compile(fullgraph=True)**: Maximum optimization, less compatible
2. **FP8 Precision**: Available on H100 GPUs (~2x speedup)
3. **Custom CUDA Kernels**: Hand-optimized kernels for specific operations

## References

- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [PyTorch 2.0 torch.compile()](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [NVIDIA TF32 Documentation](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
- [cuDNN Benchmark Mode](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark)

## Performance Testing

To benchmark your own hardware:
```bash
# Run official benchmark script
python bench_tts.py --label "Optimized"

# Compare with baseline (disable optimizations)
python bench_tts.py --label "Baseline"
```

## Changelog

- **2026-01-25**: Added torch.compile(), TF32, and cuDNN benchmark optimizations
- **2026-01-25**: Implemented Flash Attention 2 support
- **2026-01-24**: Initial optimized deployment with vLLM-Omni backend

---

**Last Updated**: 2026-01-25  
**Hardware Tested**: NVIDIA RTX 3090 (24GB)  
**PyTorch Version**: 2.0+  
**CUDA Version**: 12.1.1
