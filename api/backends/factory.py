# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import os
import logging
from typing import Optional

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend
from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None


def get_backend() -> TTSBackend:
    """
    Get or create the global TTS backend instance.
    
    The backend is selected based on the TTS_BACKEND environment variable:
    - "official" (default): Use official Qwen3-TTS implementation
    - "vllm_omni": Use vLLM-Omni for faster inference
    
    Returns:
        TTSBackend instance
    """
    global _backend_instance
    
    if _backend_instance is not None:
        return _backend_instance
    
    # Get backend type from environment
    backend_type = os.getenv("TTS_BACKEND", "official").lower()
    
    # Get model name from environment (optional override)
    model_name = os.getenv("TTS_MODEL_NAME")
    
    logger.info(f"Initializing TTS backend: {backend_type}")
    
    if backend_type == "official":
        # Official backend
        if model_name:
            _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
        else:
            # Use default CustomVoice model
            _backend_instance = OfficialQwen3TTSBackend()
        
        logger.info(f"Using official Qwen3-TTS backend with model: {_backend_instance.get_model_id()}")
    
    elif backend_type == "vllm_omni" or backend_type == "vllm-omni" or backend_type == "vllm":
        # vLLM-Omni backend
        if model_name:
            _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
        else:
            # Use 1.7B model for best quality/speed tradeoff
            _backend_instance = VLLMOmniQwen3TTSBackend(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
        
        logger.info(f"Using vLLM-Omni backend with model: {_backend_instance.get_model_id()}")
    
    else:
        logger.error(f"Unknown backend type: {backend_type}")
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. "
            f"Supported values: 'official', 'vllm_omni'"
        )
    
    return _backend_instance


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """
    Initialize the backend and optionally perform warmup.
    
    Args:
        warmup: Whether to run a warmup inference
    
    Returns:
        Initialized TTSBackend instance
    """
    backend = get_backend()
    
    # Initialize the backend
    await backend.initialize()
    
    # Perform warmup if requested
    if warmup:
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        if warmup_enabled:
            logger.info("Performing backend warmup...")
            try:
                # Run a simple warmup generation
                await backend.generate_speech(
                    text="Hello, this is a warmup test.",
                    voice="Vivian",
                    language="English",
                )
                logger.info("Backend warmup completed successfully")
            except Exception as e:
                logger.warning(f"Backend warmup failed (non-critical): {e}")
    
    return backend


def reset_backend() -> None:
    """Reset the global backend instance (useful for testing)."""
    global _backend_instance
    _backend_instance = None
