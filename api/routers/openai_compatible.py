# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import logging
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import OpenAISpeechRequest, ModelInfo, VoiceInfo
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# Global TTS model instance (lazy loaded)
_tts_model = None
_tts_model_lock = None


# Available models
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Sophia",
    "nova": "Isabella",
    "onyx": "Evan",
    "shimmer": "Lily",
}


async def get_tts_model():
    """Get the global TTS model instance, initializing if needed."""
    global _tts_model, _tts_model_lock
    import asyncio
    
    if _tts_model_lock is None:
        _tts_model_lock = asyncio.Lock()
    
    if _tts_model is None:
        async with _tts_model_lock:
            if _tts_model is None:
                try:
                    import torch
                    from qwen_tts import Qwen3TTSModel
                    
                    # Determine device
                    if torch.cuda.is_available():
                        device = "cuda:0"
                        dtype = torch.bfloat16
                    else:
                        device = "cpu"
                        dtype = torch.float32
                    
                    logger.info(f"Loading Qwen3-TTS model on {device}...")
                    
                    # Try to load the custom voice model first
                    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                    
                    _tts_model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=device,
                        dtype=dtype,
                    )
                    
                    logger.info(f"Qwen3-TTS model loaded successfully on {device}")
                    
                except Exception as e:
                    logger.error(f"Failed to load TTS model: {e}")
                    raise RuntimeError(f"Failed to initialize TTS model: {e}")
    
    return _tts_model


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using Qwen3-TTS.
    
    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    tts_model = await get_tts_model()
    
    # Map voice name
    voice_name = get_voice_name(voice)
    
    # Generate speech
    try:
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice_name,
            instruct=instruct,
        )
        
        audio = wavs[0]
        
        # Apply speed adjustment if needed
        if speed != 1.0 and LIBROSA_AVAILABLE:
            audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
        elif speed != 1.0:
            logger.warning("Speed adjustment requested but librosa not available")
        
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """
    OpenAI-compatible endpoint for text-to-speech.
    
    Generates audio from the input text using the specified voice and model.
    """
    # Validate model
    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )
    
    try:
        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)
        
        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )
        
        # Generate speech
        audio, sample_rate = await generate_speech(
            text=normalized_text,
            voice=request.voice,
            language=request.language or "Auto",
            instruct=request.instruct,
            speed=request.speed,
        )
        
        # Encode audio to requested format
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        
        # Get content type
        content_type = get_content_type(request.response_format)
        
        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    # Default voices (always available)
    default_voices = [
        VoiceInfo(id="Vivian", name="Vivian", language="English", description="Female voice"),
        VoiceInfo(id="Ryan", name="Ryan", language="English", description="Male voice"),
        VoiceInfo(id="Sophia", name="Sophia", language="English", description="Female voice"),
        VoiceInfo(id="Isabella", name="Isabella", language="English", description="Female voice"),
        VoiceInfo(id="Evan", name="Evan", language="English", description="Male voice"),
        VoiceInfo(id="Lily", name="Lily", language="English", description="Female voice"),
    ]
    
    # OpenAI-compatible voice aliases
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Sophia)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Isabella)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Evan)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Lily)"),
    ]
    
    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]
    
    try:
        tts_model = await get_tts_model()
        
        # Get supported speakers from the model
        speakers = []
        if hasattr(tts_model.model, 'get_supported_speakers'):
            raw_speakers = tts_model.model.get_supported_speakers()
            if raw_speakers:
                speakers = list(raw_speakers)
        
        # Get supported languages
        languages = []
        if hasattr(tts_model.model, 'get_supported_languages'):
            raw_languages = tts_model.model.get_supported_languages()
            if raw_languages:
                languages = list(raw_languages)
        
        # Build voice list from model if available
        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        return {
            "voices": voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }
        
    except Exception:
        # Return default voices if model is not loaded
        return {
            "voices": [v.model_dump() for v in default_voices] + [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }
