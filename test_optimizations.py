#!/usr/bin/env python3
"""
Test various PyTorch optimizations for Qwen3-TTS inference speed.

This script tests:
1. torch.compile() with different modes
2. TF32 precision  
3. cuDNN benchmark mode
4. SDPA backend selection
5. Different torch.compile modes (reduce-overhead, max-autotune)
"""

import time
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# Test texts of varying lengths
TEST_CASES = [
    ("Hello world!", "Short"),
    ("The quick brown fox jumps over the lazy dog.", "Sentence"),
    ("Artificial intelligence is transforming the way we live and work. From healthcare to transportation, AI is making our lives easier and more efficient.", "Medium"),
    ("In recent years, artificial intelligence has made remarkable progress across many domains. Machine learning algorithms can now recognize images, understand natural language, and even generate creative content like music and art. As these technologies continue to advance, they promise to revolutionize industries and create new opportunities for innovation.", "Long"),
]


def benchmark_model(model, test_cases, label="Baseline", warmup=1):
    """Benchmark model inference speed."""
    results = []
    
    # Warmup
    print(f"\n[{label}] Warming up with {warmup} iterations...")
    for _ in range(warmup):
        _, _ = model.generate_custom_voice(
            text="Warmup test.",
            language="English", 
            speaker="Vivian",
        )
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"\n[{label}] Running benchmarks...")
    for text, name in test_cases:
        # Time the generation
        start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Vivian",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU operations complete
        elapsed = time.time() - start
        
        # Calculate Real-Time Factor
        audio_duration = len(wavs[0]) / sr
        rtf = elapsed / audio_duration
        
        word_count = len(text.split())
        results.append({
            'name': name,
            'text': text,
            'words': word_count,
            'latency': elapsed,
            'rtf': rtf,
            'audio_duration': audio_duration,
        })
        
        print(f"  {name:10s} ({word_count:2d}w): {elapsed:6.2f}s (RTF: {rtf:.2f}, Audio: {audio_duration:.2f}s)")
    
    # Calculate average
    avg_latency = np.mean([r['latency'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])
    
    print(f"\n[{label}] Average: {avg_latency:.2f}s latency, RTF {avg_rtf:.2f}")
    
    return results, avg_latency, avg_rtf


def test_baseline(model_name):
    """Test baseline performance (Flash Attention 2, no torch.compile)."""
    print("\n" + "="*80)
    print("TEST 1: BASELINE (Flash Attention 2, no torch.compile)")
    print("="*80)
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    results, avg_lat, avg_rtf = benchmark_model(model, TEST_CASES, "Baseline")
    
    del model
    torch.cuda.empty_cache()
    
    return results, avg_lat, avg_rtf


def test_torch_compile_reduce_overhead(model_name):
    """Test torch.compile() with reduce-overhead mode."""
    print("\n" + "="*80)
    print("TEST 2: torch.compile(mode='reduce-overhead')")
    print("="*80)
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    print("Compiling model with reduce-overhead mode...")
    model.model = torch.compile(
        model.model,
        mode="reduce-overhead",
        fullgraph=False,
    )
    
    results, avg_lat, avg_rtf = benchmark_model(model, TEST_CASES, "Compile-ReduceOverhead", warmup=2)
    
    del model
    torch.cuda.empty_cache()
    
    return results, avg_lat, avg_rtf


def test_torch_compile_max_autotune(model_name):
    """Test torch.compile() with max-autotune mode."""
    print("\n" + "="*80)
    print("TEST 3: torch.compile(mode='max-autotune')")
    print("="*80)
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    print("Compiling model with max-autotune mode...")
    model.model = torch.compile(
        model.model,
        mode="max-autotune",
        fullgraph=False,
    )
    
    results, avg_lat, avg_rtf = benchmark_model(model, TEST_CASES, "Compile-MaxAutotune", warmup=2)
    
    del model
    torch.cuda.empty_cache()
    
    return results, avg_lat, avg_rtf


def test_cudnn_benchmark(model_name):
    """Test with cuDNN benchmark enabled."""
    print("\n" + "="*80)
    print("TEST 4: cuDNN Benchmark + TF32")
    print("="*80)
    
    # Enable cuDNN benchmark and TF32
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    results, avg_lat, avg_rtf = benchmark_model(model, TEST_CASES, "cuDNN+TF32")
    
    del model
    torch.cuda.empty_cache()
    
    return results, avg_lat, avg_rtf


def test_all_optimizations(model_name):
    """Test with all optimizations combined."""
    print("\n" + "="*80)
    print("TEST 5: ALL OPTIMIZATIONS (Compile + cuDNN + TF32)")
    print("="*80)
    
    # Enable all optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    print("Compiling model with reduce-overhead mode...")
    model.model = torch.compile(
        model.model,
        mode="reduce-overhead",
        fullgraph=False,
    )
    
    results, avg_lat, avg_rtf = benchmark_model(model, TEST_CASES, "ALL-OPTIMIZATIONS", warmup=2)
    
    del model
    torch.cuda.empty_cache()
    
    return results, avg_lat, avg_rtf


def main():
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    print("="*80)
    print("QWEN3-TTS OPTIMIZATION BENCHMARK")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: CUDA (GPU)")
    print(f"Precision: BFloat16")
    print(f"Attention: Flash Attention 2")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    all_results = {}
    
    try:
        # Test 1: Baseline (Flash Attention 2)
        results, avg_lat, avg_rtf = test_baseline(model_name)
        all_results['Baseline (Flash Attn 2)'] = {'results': results, 'avg_lat': avg_lat, 'avg_rtf': avg_rtf}
        
        # Test 2: torch.compile with reduce-overhead
        results, avg_lat, avg_rtf = test_torch_compile_reduce_overhead(model_name)
        all_results['torch.compile (reduce-overhead)'] = {'results': results, 'avg_lat': avg_lat, 'avg_rtf': avg_rtf}
        
        # Test 3: torch.compile with max-autotune
        results, avg_lat, avg_rtf = test_torch_compile_max_autotune(model_name)
        all_results['torch.compile (max-autotune)'] = {'results': results, 'avg_lat': avg_lat, 'avg_rtf': avg_rtf}
        
        # Test 4: cuDNN benchmark + TF32
        results, avg_lat, avg_rtf = test_cudnn_benchmark(model_name)
        all_results['cuDNN Benchmark + TF32'] = {'results': results, 'avg_lat': avg_lat, 'avg_rtf': avg_rtf}
        
        # Test 5: All optimizations combined
        results, avg_lat, avg_rtf = test_all_optimizations(model_name)
        all_results['ALL Optimizations'] = {'results': results, 'avg_lat': avg_lat, 'avg_rtf': avg_rtf}
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION BENCHMARK SUMMARY")
    print("="*80)
    
    baseline_rtf = all_results['Baseline (Flash Attn 2)']['avg_rtf']
    baseline_lat = all_results['Baseline (Flash Attn 2)']['avg_lat']
    
    print(f"\n{'Configuration':<35s} │ Avg RTF │ Avg Latency │ vs Baseline")
    print("─" * 80)
    
    for config_name, data in all_results.items():
        avg_rtf = data['avg_rtf']
        avg_lat = data['avg_lat']
        speedup_pct = ((baseline_rtf - avg_rtf) / baseline_rtf) * 100
        
        marker = "🏆" if avg_rtf == min(d['avg_rtf'] for d in all_results.values()) else "  "
        
        print(f"{marker} {config_name:<33s} │  {avg_rtf:5.2f}  │   {avg_lat:6.2f}s   │ {speedup_pct:+6.1f}%")
    
    # Find best configuration
    best_config = min(all_results.items(), key=lambda x: x[1]['avg_rtf'])
    print("\n" + "="*80)
    print(f"🏆 BEST CONFIGURATION: {best_config[0]}")
    print(f"   Average RTF: {best_config[1]['avg_rtf']:.2f}")
    print(f"   Average Latency: {best_config[1]['avg_lat']:.2f}s")
    improvement = ((baseline_rtf - best_config[1]['avg_rtf']) / baseline_rtf) * 100
    print(f"   Improvement over baseline: {improvement:.1f}%")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
