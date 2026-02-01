# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Official Qwen3-TTS backend implementation.

This backend uses the official Qwen3-TTS Python implementation
from the qwen_tts package. Supports both CustomVoice and Base models.

- CustomVoice model: Uses built-in premium voices
- Base model: Supports voice cloning from reference audio samples
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Directory for custom voice samples (mounted in container)
VOICE_SAMPLES_DIR = os.environ.get("VOICE_SAMPLES_DIR", "/app/voice-samples")


class OfficialQwen3TTSBackend(TTSBackend):
    """Official Qwen3-TTS backend using the qwen_tts package."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        """
        Initialize the official backend.
        
        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__()
        self.model_name = model_name
        self._ready = False
        self._voice_prompts_cache: Dict[str, Any] = {}  # Cache for voice clone prompts
        self._custom_voices: Dict[str, dict] = {}  # Map of voice name -> {path, transcript}
        self._is_base_model = "Base" in model_name
    
    def _scan_voice_samples(self) -> None:
        """Scan the voice samples directory and build voice mapping."""
        self._custom_voices = {}
        
        if not os.path.exists(VOICE_SAMPLES_DIR):
            logger.info(f"Voice samples directory not found: {VOICE_SAMPLES_DIR}")
            return
        
        # Scan for .wav and .mp3 files
        samples_path = Path(VOICE_SAMPLES_DIR)
        for ext in ["*.wav", "*.mp3"]:
            for audio_file in samples_path.glob(ext):
                # Extract voice name from filename (just the stem without extension)
                # e.g., "Jarvis.wav" -> "Jarvis", "General_Joe.wav" -> "General_Joe"
                voice_name = audio_file.stem
                
                # Create variations for flexible matching
                voice_key = voice_name  # e.g., "General_Joe"
                voice_display = voice_name.replace("_", " ")  # e.g., "General Joe"
                
                # Store mapping (prefer .wav over .mp3 if both exist)
                if voice_key not in self._custom_voices or audio_file.suffix == ".wav":
                    voice_info = {
                        "path": str(audio_file),
                        "name": voice_display,
                        "key": voice_key,
                    }
                    # Add multiple keys for flexible matching
                    self._custom_voices[voice_key] = voice_info
                    self._custom_voices[voice_key.lower()] = voice_info
                    self._custom_voices[voice_display] = voice_info
                    self._custom_voices[voice_display.lower()] = voice_info
        
        if self._custom_voices:
            unique_voices = set(v["name"] for v in self._custom_voices.values())
            logger.info(f"Loaded {len(unique_voices)} custom voice samples from {VOICE_SAMPLES_DIR}")
            for voice in sorted(unique_voices):
                logger.info(f"  - {voice}")
    
    def _get_custom_voice(self, voice_name: str) -> Optional[dict]:
        """Get custom voice info by name (case-insensitive)."""
        # Try exact match first
        if voice_name in self._custom_voices:
            return self._custom_voices[voice_name]
        
        # Try lowercase
        voice_lower = voice_name.lower()
        if voice_lower in self._custom_voices:
            return self._custom_voices[voice_lower]
        
        # Try with underscores instead of spaces
        voice_underscore = voice_name.replace(" ", "_")
        if voice_underscore in self._custom_voices:
            return self._custom_voices[voice_underscore]
        
        # Try partial match (first name only)
        first_name = voice_name.split()[0] if " " in voice_name else voice_name
        if first_name.lower() in self._custom_voices:
            return self._custom_voices[first_name.lower()]
        
        return None
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("Official backend already initialized")
            return
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            
            logger.info(f"Loading Qwen3-TTS model '{self.model_name}' on {self.device}...")
            logger.info(f"Model type: {'Base (voice cloning)' if self._is_base_model else 'CustomVoice (built-in voices)'}")
            
            # Detect available attention implementation
            attn_impl = "sdpa"  # Default to PyTorch's native scaled dot-product attention
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 is available, using flash_attention_2")
            except ImportError:
                logger.info("Flash Attention 2 not available, using SDPA (still fast, just uses more VRAM)")
            
            # Load model with detected attention implementation
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            
            # Scan for custom voice samples
            self._scan_voice_samples()
            
            # Apply torch.compile() optimization for faster inference
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                logger.info("Applying torch.compile() optimization...")
                try:
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    logger.info("torch.compile() optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile(): {e}")
            
            # Enable cuDNN benchmarking for optimal convolution algorithms
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
            
            # Enable TF32 for faster matmul on Ampere+ GPUs
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster matmul")
            
            self._ready = True
            logger.info(f"Official Qwen3-TTS backend loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load official TTS backend: {e}")
            raise RuntimeError(f"Failed to initialize official TTS backend: {e}")
    
    async def _generate_with_voice_clone(
        self,
        text: str,
        voice_info: dict,
        language: str = "English",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using voice cloning."""
        voice_name = voice_info["name"]
        voice_path = voice_info["path"]
        
        logger.info(f"Using voice cloning with sample: {voice_name}")
        
        # Check if we have a cached voice prompt
        cache_key = voice_path
        if cache_key not in self._voice_prompts_cache:
            logger.info(f"Creating voice clone prompt for: {voice_name}")
            # Create voice clone prompt (x_vector_only_mode=True means no transcript needed)
            try:
                self._voice_prompts_cache[cache_key] = self.model.create_voice_clone_prompt(
                    ref_audio=voice_path,
                    x_vector_only_mode=True,  # Don't require transcript
                )
            except Exception as e:
                logger.error(f"Failed to create voice clone prompt: {e}")
                raise
        
        # Generate speech with cached voice prompt
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=self._voice_prompts_cache[cache_key],
        )
        
        audio = wavs[0]
        
        # Apply speed adjustment if needed
        if speed != 1.0 and LIBROSA_AVAILABLE:
            audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
        
        return audio, sr
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        If voice matches a custom sample, uses voice cloning.
        Otherwise uses built-in voices (CustomVoice model) or fails (Base model).
        """
        if not self._ready:
            await self.initialize()
        
        try:
            # Check if this is a custom voice
            custom_voice = self._get_custom_voice(voice)
            
            if custom_voice:
                # Use voice cloning
                return await self._generate_with_voice_clone(
                    text=text,
                    voice_info=custom_voice,
                    language=language if language != "Auto" else "English",
                    speed=speed,
                )
            
            # Not a custom voice - use built-in voices
            if self._is_base_model:
                # Base model doesn't have built-in named voices
                # List available custom voices
                available = list(set(v["name"] for v in self._custom_voices.values()))
                logger.error(f"Voice '{voice}' not found. Available custom voices: {available}")
                raise ValueError(
                    f"Voice '{voice}' not found. Using Base model which requires custom voice samples. "
                    f"Available voices: {available}"
                )
            
            # CustomVoice model - use built-in voices
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct,
            )
            
            audio = wavs[0]
            
            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "official"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        voices = []
        
        # Add custom voices
        custom = list(set(v["name"] for v in self._custom_voices.values()))
        voices.extend(sorted(custom))
        
        # Add built-in voices if using CustomVoice model
        if not self._is_base_model:
            builtin = ["Vivian", "Ryan", "Serena", "Aiden", "Dylan", "Eric", 
                      "Uncle_Fu", "Ono_Anna", "Sohee"]
            voices.extend(builtin)
        
        return voices
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian", "Auto"]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    async def unload(self) -> None:
        """
        Unload the model from GPU memory to free VRAM.
        
        This properly cleans up CUDA memory and allows other applications
        to use the GPU. The model will be reloaded on the next request.
        """
        if not self._ready:
            logger.info("Backend not loaded, nothing to unload")
            return
        
        logger.info("Unloading Qwen3-TTS model from GPU memory...")
        
        try:
            import torch
            import gc
            
            # Clear the model
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clear voice prompts cache (contains tensor data)
            if self._voice_prompts_cache:
                cache_size = len(self._voice_prompts_cache)
                self._voice_prompts_cache.clear()
                logger.info(f"Cleared {cache_size} cached voice prompts")
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA memory cache cleared")
            
            self._ready = False
            logger.info("Model unloaded successfully - VRAM freed")
            
        except Exception as e:
            logger.error(f"Error during model unload: {e}")
            # Still mark as not ready even if cleanup had issues
            self._ready = False
            self.model = None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
            "model_type": "Base (voice cloning)" if self._is_base_model else "CustomVoice",
            "custom_voices_count": len(set(v["name"] for v in self._custom_voices.values())),
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["gpu_available"] = True
                if torch.cuda.current_device() >= 0:
                    device_idx = torch.cuda.current_device()
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                    
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"
                    
                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        return info
