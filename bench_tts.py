#!/usr/bin/env python3
"""
TTS Benchmark Script for Qwen3-TTS

Measures:
- Cold start (first request after server start)
- Warm runs (median of N runs)
- RTF (Real-Time Factor) = wall_time / audio_duration

Usage:
    # Benchmark official backend
    docker compose down
    docker compose up -d --build qwen3-tts-gpu
    python3 bench_tts.py | tee bench_official.txt
    
    # Benchmark vLLM-Omni backend
    docker compose down
    docker compose --profile vllm up -d --build qwen3-tts-vllm
    python3 bench_tts.py | tee bench_vllm.txt
"""

import io
import sys
import time
import json
import argparse
import statistics as stats
from pathlib import Path
from datetime import datetime

import requests
import soundfile as sf


# Test prompts of varying lengths
PROMPTS = [
    ("2 words", "Hello world"),
    ("short sentence", "Kia ora koutou, welcome to today's meeting."),
    ("medium paragraph", 
     "The quick brown fox jumps over the lazy dog near the riverbank. "
     "This is a test of text-to-speech generation quality."),
    ("long paragraph",
     "Artificial intelligence has revolutionized the way we interact with technology. "
     "Text-to-speech technology has advanced significantly in recent years. "
     "Modern neural networks can generate remarkably natural-sounding speech. "
     "The Qwen3-TTS model represents the latest breakthrough in this field."),
]


def post_tts(base_url: str, text: str, voice: str = "Vivian") -> tuple[bytes, float]:
    """
    Make a TTS request and return audio bytes with timing.
    
    Args:
        base_url: Server base URL
        text: Text to synthesize
        voice: Voice/speaker name
        
    Returns:
        Tuple of (wav_bytes, elapsed_seconds)
    """
    url = f"{base_url}/v1/audio/speech"
    payload = {
        "model": "qwen3-tts",
        "voice": voice,
        "input": text,
        "response_format": "wav",
    }
    
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    dt = time.time() - t0
    
    return r.content, dt


def wav_duration_s(wav_bytes: bytes) -> float:
    """
    Calculate duration of WAV audio in seconds.
    
    Args:
        wav_bytes: WAV file bytes
        
    Returns:
        Duration in seconds
    """
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        n = data.shape[0]
    else:
        n = len(data)
    return n / float(sr)


def get_health(base_url: str) -> dict:
    """Get server health status."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def bench(
    base_url: str, 
    label: str, 
    warm_runs: int = 5,
    output_dir: Path = None
) -> dict:
    """
    Run benchmark and return results.
    
    Args:
        base_url: Server base URL
        label: Label for this benchmark run
        warm_runs: Number of warm runs per prompt
        output_dir: Directory to save audio files (optional)
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"Server: {base_url}")
    print(f"Warm runs: {warm_runs}")
    print(f"{'='*70}")
    
    # Check server health
    health = get_health(base_url)
    print(f"\nServer health: {health.get('status', 'unknown')}")
    if 'backend' in health:
        print(f"Backend: {health['backend'].get('name', 'unknown')}")
        print(f"Model: {health['backend'].get('model_id', 'unknown')}")
    
    results = {
        "label": label,
        "base_url": base_url,
        "timestamp": datetime.now().isoformat(),
        "health": health,
        "prompts": []
    }
    
    for name, text in PROMPTS:
        word_count = len(text.split())
        char_count = len(text)
        
        print(f"\n[{name}] ({word_count} words, {char_count} chars)")
        print(f"  Text: {text[:50]}..." if len(text) > 50 else f"  Text: {text}")
        
        prompt_result = {
            "name": name,
            "text": text,
            "word_count": word_count,
            "char_count": char_count,
        }
        
        try:
            # Cold run (first request)
            wav, cold_t = post_tts(base_url, text)
            dur = wav_duration_s(wav)
            cold_rtf = cold_t / dur if dur > 0 else float("inf")
            
            prompt_result["cold"] = {
                "time_s": round(cold_t, 3),
                "audio_duration_s": round(dur, 3),
                "rtf": round(cold_rtf, 3),
            }
            
            print(f"  Cold: {cold_t:.2f}s | audio: {dur:.2f}s | RTF: {cold_rtf:.2f}")
            
            # Save cold audio if output_dir specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{name.replace(' ', '_')}_cold.wav"
                (output_dir / fname).write_bytes(wav)
            
            # Warm runs
            times = []
            rtfs = []
            
            for i in range(warm_runs):
                wav, dt = post_tts(base_url, text)
                d = wav_duration_s(wav)
                times.append(dt)
                rtfs.append(dt / d if d > 0 else float("inf"))
                print(f"    Run {i+1}: {dt:.2f}s | RTF: {rtfs[-1]:.2f}")
            
            median_time = stats.median(times)
            median_rtf = stats.median(rtfs)
            min_time = min(times)
            max_time = max(times)
            p95_time = stats.quantiles(times, n=20)[-1] if len(times) >= 2 else max_time
            
            prompt_result["warm"] = {
                "runs": warm_runs,
                "median_time_s": round(median_time, 3),
                "min_time_s": round(min_time, 3),
                "max_time_s": round(max_time, 3),
                "p95_time_s": round(p95_time, 3),
                "median_rtf": round(median_rtf, 3),
            }
            
            print(f"  Warm median: {median_time:.2f}s | RTF: {median_rtf:.2f}")
            print(f"  Warm p95: {p95_time:.2f}s | min: {min_time:.2f}s | max: {max_time:.2f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            prompt_result["error"] = str(e)
        
        results["prompts"].append(prompt_result)
    
    # Summary
    print(f"\n{'-'*70}")
    print("SUMMARY")
    print(f"{'-'*70}")
    
    successful = [p for p in results["prompts"] if "warm" in p]
    if successful:
        all_medians = [p["warm"]["median_time_s"] for p in successful]
        all_rtfs = [p["warm"]["median_rtf"] for p in successful]
        
        avg_time = sum(all_medians) / len(all_medians)
        avg_rtf = sum(all_rtfs) / len(all_rtfs)
        
        results["summary"] = {
            "successful_prompts": len(successful),
            "total_prompts": len(PROMPTS),
            "avg_warm_median_time_s": round(avg_time, 3),
            "avg_warm_median_rtf": round(avg_rtf, 3),
        }
        
        print(f"Successful: {len(successful)}/{len(PROMPTS)} prompts")
        print(f"Avg warm median: {avg_time:.2f}s")
        print(f"Avg warm RTF: {avg_rtf:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark Script")
    parser.add_argument(
        "--url", 
        default="http://localhost:8880",
        help="Server base URL (default: http://localhost:8880)"
    )
    parser.add_argument(
        "--label",
        default="CURRENT_BACKEND",
        help="Label for this benchmark run"
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=5,
        help="Number of warm runs per prompt (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save audio files"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = bench(
        base_url=args.url,
        label=args.label,
        warm_runs=args.warm_runs,
        output_dir=args.output_dir,
    )
    
    # Save to JSON if requested
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
