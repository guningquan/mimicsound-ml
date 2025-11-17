"""
Audio utilities for MimicSound audio modal integration.
This module provides functions for generating random raw audio
and handling audio data in debug mode.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


def get_audio_shape() -> Tuple[int]:
    """
    Get the expected shape for raw audio.
    
    Returns:
        Tuple of (audio_samples,) for 2-second audio at 16kHz
        Shape: (32000,) - 32000 samples for 2 seconds at 16kHz
    """
    return (32000,)


def generate_random_raw_audio(
    batch_size: int = 1,
    duration: float = 2.0,
    sample_rate: int = 16000,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate random raw audio for debugging purposes.
    
    This function creates synthetic raw audio that mimics the structure
    of real audio data, useful for testing the audio modal integration.
    
    Args:
        batch_size: Number of audio samples to generate
        duration: Duration in seconds (default: 2.0)
        sample_rate: Sample rate in Hz (default: 16000)
        device: Device to place the tensor on
    
    Returns:
        Random raw audio tensor of shape [batch_size, audio_samples]
    """
    # Calculate number of audio samples
    audio_samples = int(duration * sample_rate)
    
    # Generate random raw audio
    # Use normal distribution with small std to mimic normalized audio
    raw_audio = torch.randn(batch_size, audio_samples) * 0.1
    
    # Add some structure to make it more realistic
    # Add sinusoidal components to simulate speech-like patterns
    t = torch.linspace(0, duration, audio_samples)
    
    for i in range(batch_size):
        # Add multiple frequency components
        audio = torch.zeros(audio_samples)
        
        # Add fundamental frequency (speech-like)
        audio += 0.3 * torch.sin(2 * np.pi * 200 * t)  # 200 Hz fundamental
        
        # Add harmonics
        audio += 0.2 * torch.sin(2 * np.pi * 400 * t)  # 400 Hz harmonic
        audio += 0.1 * torch.sin(2 * np.pi * 600 * t)  # 600 Hz harmonic
        
        # Add some noise
        audio += torch.randn(audio_samples) * 0.05
        
        # Apply envelope to make it more speech-like
        envelope = torch.exp(-t * 0.5) * (1 + 0.5 * torch.sin(2 * np.pi * 2 * t))
        audio *= envelope
        
        raw_audio[i] = audio
    
    # Move to specified device
    if device is not None:
        raw_audio = raw_audio.to(device)
    
    logger.debug(f"Generated random raw audio with shape: {raw_audio.shape}")
    
    return raw_audio


def generate_audio_data_for_batch(
    batch_size: int,
    seq_length: int,
    device: Optional[torch.device] = None,
    debug_mode: bool = True
) -> torch.Tensor:
    """
    Generate audio data for a batch of sequences.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        device: Device to place tensors on
        debug_mode: Whether to generate random data (True) or load real data (False)
    
    Returns:
        Audio data tensor of shape [batch_size, seq_length, audio_samples]
    """
    if debug_mode:
        # Generate random raw audio for each timestep
        audio_data = []
        for _ in range(seq_length):
            raw_audio = generate_random_raw_audio(
                batch_size=batch_size,
                device=device
            )
            audio_data.append(raw_audio)
        
        # Stack along sequence dimension
        audio_data = torch.stack(audio_data, dim=1)  # [batch_size, seq_length, audio_samples]
        
        logger.debug(f"Generated audio batch with shape: {audio_data.shape}")
        return audio_data
    else:
        # TODO: Implement real audio data loading
        # For now, fall back to random generation
        logger.warning("Real audio data loading not implemented, using random data")
        return generate_audio_data_for_batch(batch_size, seq_length, device, debug_mode=True)


def validate_audio_shape(audio_tensor: torch.Tensor) -> bool:
    """
    Validate that the audio tensor has the expected shape.
    
    Args:
        audio_tensor: Audio tensor to validate
    
    Returns:
        True if shape is valid, False otherwise
    """
    expected_shape = get_audio_shape()
    
    if audio_tensor.dim() == 3:  # [batch, seq, audio_samples]
        actual_shape = audio_tensor.shape[-1:]
    elif audio_tensor.dim() == 2:  # [batch, audio_samples]
        actual_shape = audio_tensor.shape[-1:]
    else:
        logger.error(f"Invalid audio tensor dimensions: {audio_tensor.dim()}")
        return False
    
    if actual_shape != expected_shape:
        logger.error(f"Invalid audio shape: expected {expected_shape}, got {actual_shape}")
        return False
    
    return True


def preprocess_audio_data(
    audio_tensor: torch.Tensor,
    normalize: bool = True,
    log_scale: bool = True
) -> torch.Tensor:
    """
    Preprocess audio data for model input.
    
    Args:
        audio_tensor: Input audio tensor
        normalize: Whether to normalize the data
        log_scale: Whether to apply log scaling
    
    Returns:
        Preprocessed audio tensor
    """
    processed_audio = audio_tensor.clone()
    
    # Apply log scaling
    if log_scale:
        processed_audio = torch.log(processed_audio + 1e-8)
    
    # Normalize
    if normalize:
        # Normalize across the batch and sequence dimensions
        mean = processed_audio.mean(dim=(0, 1), keepdim=True)
        std = processed_audio.std(dim=(0, 1), keepdim=True) + 1e-8
        processed_audio = (processed_audio - mean) / std
    
    return processed_audio


def get_audio_feature_dim() -> int:
    """
    Get the output feature dimension for the AST encoder.
    
    Returns:
        Feature dimension (512)
    """
    return 512


# Audio configuration constants
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'duration': 2.0,
    'audio_samples': 32000,  # 2 seconds at 16kHz
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 128,
    'feature_dim': 512,
    'patch_size': 16,
}


def get_audio_config() -> dict:
    """
    Get audio configuration dictionary.
    
    Returns:
        Dictionary containing audio processing parameters
    """
    return AUDIO_CONFIG.copy()
