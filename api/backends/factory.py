# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import asyncio
import os
import logging
import time
from typing import Optional

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend
from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None

# Activity tracking for auto-unload
_last_activity_time: float = 0.0
_auto_unload_task: Optional[asyncio.Task] = None
_auto_unload_enabled: bool = False

# Configuration
INACTIVITY_TIMEOUT_MINUTES = int(os.getenv("TTS_INACTIVITY_TIMEOUT_MINUTES", "0"))  # 0 = disabled


def update_activity() -> None:
    """Update the last activity timestamp. Called on each TTS request."""
    global _last_activity_time
    _last_activity_time = time.time()


def get_inactivity_seconds() -> float:
    """Get seconds since last activity."""
    if _last_activity_time == 0:
        return 0
    return time.time() - _last_activity_time


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


async def unload_backend() -> bool:
    """
    Unload the TTS backend to free GPU VRAM.
    
    Returns:
        True if unloaded, False if not loaded or failed
    """
    global _backend_instance, _last_activity_time
    
    if _backend_instance is None:
        logger.info("No backend loaded, nothing to unload")
        return False
    
    if not _backend_instance.is_ready():
        logger.info("Backend not ready, nothing to unload")
        return False
    
    try:
        await _backend_instance.unload()
        _last_activity_time = 0.0  # Reset activity time
        logger.info("Backend unloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to unload backend: {e}")
        return False


async def _auto_unload_checker() -> None:
    """Background task that checks for inactivity and unloads the model."""
    global _backend_instance, _auto_unload_enabled
    
    timeout_seconds = INACTIVITY_TIMEOUT_MINUTES * 60
    check_interval = min(60, timeout_seconds / 4)  # Check at least every minute
    
    logger.info(f"Auto-unload checker started (timeout: {INACTIVITY_TIMEOUT_MINUTES} minutes)")
    
    while _auto_unload_enabled:
        try:
            await asyncio.sleep(check_interval)
            
            # Skip if no backend or not ready
            if _backend_instance is None or not _backend_instance.is_ready():
                continue
            
            # Skip if no activity yet (model just loaded, no requests)
            if _last_activity_time == 0:
                continue
            
            inactivity = get_inactivity_seconds()
            
            if inactivity >= timeout_seconds:
                logger.info(
                    f"Model inactive for {inactivity/60:.1f} minutes "
                    f"(threshold: {INACTIVITY_TIMEOUT_MINUTES} minutes) - unloading to free VRAM"
                )
                await unload_backend()
            else:
                remaining = (timeout_seconds - inactivity) / 60
                logger.debug(f"Model active - will unload in {remaining:.1f} minutes if idle")
                
        except asyncio.CancelledError:
            logger.info("Auto-unload checker cancelled")
            break
        except Exception as e:
            logger.error(f"Error in auto-unload checker: {e}")
            await asyncio.sleep(10)  # Wait a bit before retrying


async def start_auto_unload() -> None:
    """Start the auto-unload background task if configured."""
    global _auto_unload_task, _auto_unload_enabled
    
    if INACTIVITY_TIMEOUT_MINUTES <= 0:
        logger.info("Auto-unload disabled (TTS_INACTIVITY_TIMEOUT_MINUTES not set or 0)")
        return
    
    if _auto_unload_task is not None and not _auto_unload_task.done():
        logger.info("Auto-unload checker already running")
        return
    
    _auto_unload_enabled = True
    _auto_unload_task = asyncio.create_task(_auto_unload_checker())
    logger.info(f"Auto-unload enabled: model will unload after {INACTIVITY_TIMEOUT_MINUTES} minutes of inactivity")


async def stop_auto_unload() -> None:
    """Stop the auto-unload background task."""
    global _auto_unload_task, _auto_unload_enabled
    
    _auto_unload_enabled = False
    
    if _auto_unload_task is not None:
        _auto_unload_task.cancel()
        try:
            await _auto_unload_task
        except asyncio.CancelledError:
            pass
        _auto_unload_task = None
        logger.info("Auto-unload checker stopped")
