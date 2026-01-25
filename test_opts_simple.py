#!/usr/bin/env python3
"""
Simple optimization test for Qwen3-TTS using the running API.
Tests different optimization configurations via the API endpoint.
"""

import time
import requests
import base64
import numpy as np
import soundfile as sf
from io import BytesIO

API_URL = "http://localhost:8880/v1/audio/speech"

TEST_CASES = [
    ("Hello world!", "Short"),
    ("The quick brown fox jumps over the lazy dog.", "Sentence"),
    ("Artificial intelligence is transforming the way we live and work. From healthcare to transportation, AI is making our lives easier and more efficient.", "Medium"),
    ("In recent years, artificial intelligence has made remarkable progress across many domains. Machine learning algorithms can now recognize images, understand natural language, and even generate creative content like music and art.", "Long"),
]


def benchmark_api(test_cases, label="API"):
    """Benchmark TTS API inference speed."""
    results = []
    
    print(f"\n[{label}] Running benchmarks...")
    for text, name in test_cases:
        # Time the API request
        start = time.time()
        
        response = requests.post(
            API_URL,
            json={
                "input": text,
                "voice": "Vivian",
                "model": "tts-1",
                "response_format": "mp3",
            }
        )
        
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"  {name:10s} - ERROR: {response.status_code}")
            continue
        
        # Decode audio to get duration
        audio_data = response.content
        audio, sr = sf.read(BytesIO(audio_data))
        audio_duration = len(audio) / sr
        
        # Calculate Real-Time Factor
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
    if results:
        avg_latency = np.mean([r['latency'] for r in results])
        avg_rtf = np.mean([r['rtf'] for r in results])
        print(f"\n[{label}] Average: {avg_latency:.2f}s latency, RTF {avg_rtf:.2f}")
        return results, avg_latency, avg_rtf
    else:
        return [], 0, 0


def main():
    print("="*80)
    print("QWEN3-TTS API OPTIMIZATION BENCHMARK")
    print("="*80)
    print(f"API URL: {API_URL}")
    print("Testing with Flash Attention 2 + torch.compile optimizations")
    print("="*80)
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8880/health")
            if response.status_code == 200:
                print("✓ API is ready!")
                break
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"  Waiting... ({i}s)")
    else:
        print("❌ API did not become ready in time")
        return 1
    
    # Run benchmark
    results, avg_lat, avg_rtf = benchmark_api(TEST_CASES, "Optimized-API")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\nConfiguration: Official Backend with Flash Attention 2 + torch.compile")
    print(f"Average RTF: {avg_rtf:.2f}")
    print(f"Average Latency: {avg_lat:.2f}s")
    
    # Compare with previous baseline
    baseline_rtf = 0.87  # From Flash Attn 2 benchmarks
    if avg_rtf < baseline_rtf:
        improvement = ((baseline_rtf - avg_rtf) / baseline_rtf) * 100
        print(f"Improvement over Flash Attn 2 baseline: {improvement:.1f}% faster ✓")
    else:
        degradation = ((avg_rtf - baseline_rtf) / baseline_rtf) * 100
        print(f"Performance vs Flash Attn 2 baseline: {degradation:.1f}% slower")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
