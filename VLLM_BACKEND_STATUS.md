# vLLM-Omni Backend for Qwen3-TTS

## Summary

The vLLM-Omni backend provides optimized inference for Qwen3-TTS using the [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/) framework.

## Requirements

- **Python 3.12** (required by vLLM-Omni)
- **CUDA GPU** with compute capability
- **vLLM-Omni** package (`vllm_omni`)

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run vLLM-Omni backend
docker compose --profile vllm up -d --build qwen3-tts-vllm

# Check health
curl -s http://localhost:8880/health | jq .

# Run benchmark
python3 bench_tts.py --label "vLLM-Omni" | tee bench_vllm.txt
```

### Manual Installation

```bash
# Requires Python 3.12
pip install vllm-omni soundfile numpy
```

## API Usage

The vLLM-Omni backend uses the correct import structure:

```python
from vllm import SamplingParams
from vllm_omni import Omni  # Note: vllm_omni, not vllm!

omni = Omni(model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

inputs = {
    "prompt": "<|im_start|>assistant\nHello world<|im_end|>\n<|im_start|>assistant\n",
    "additional_information": {
        "task_type": ["CustomVoice"],
        "text": ["Hello world"],
        "instruct": [""],
        "language": ["English"],
        "speaker": ["Vivian"],
        "max_new_tokens": [2048],
    },
}

for stage_outputs in omni.generate(inputs, sampling_params_list):
    for output in stage_outputs.request_output:
        audio = output.multimodal_output["audio"]
        sr = output.multimodal_output["sr"]
```

## Supported Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | ⚡⚡ | ⭐⭐⭐⭐ (Recommended) |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | ⚡⚡ | ⭐⭐⭐⭐ |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | ⚡⚡ | ⭐⭐⭐⭐ |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | ⚡⚡⚡ | ⭐⭐⭐ |

## References

- [vLLM-Omni Qwen3-TTS Example](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts/)
- [vLLM-Omni Installation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/)
- [vLLM-Omni Quickstart](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/)

---

*Updated: January 25, 2026*
