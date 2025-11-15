"""
Audio Spectrogram Transformer (AST) implementation for mimicsonic.
This module provides audio encoding capabilities for raw audio signals.
Uses transformers library's pretrained AST model for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from typing import Tuple, Optional
from robomimic.models.obs_core import EncoderCore
from robomimic.utils.obs_utils import Modality

# Import transformers AST components
try:
    from transformers import ASTModel, ASTFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Please install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# Import DoRA components
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from peft.tuners.lora import Linear as LoraLinear
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT library not available. Please install with: pip install peft")
    PEFT_AVAILABLE = False


def apply_dora_to_ast(ast_model, target_modules=None, r=8, alpha=16, dropout=0.1):
    """
    Apply DoRA (LoRA) to AST model for efficient finetuning.
    
    Args:
        ast_model: AST model from transformers
        target_modules: List of module names to apply LoRA to
        r: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
    
    Returns:
        AST model with DoRA applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for DoRA. Please install with: pip install peft")
    
    if target_modules is None:
        target_modules = [
            "query", "key", "value", "output.dense", 
            "intermediate.dense", "attention.output.dense"
        ]
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to the model
    ast_model_with_lora = get_peft_model(ast_model, lora_config)
    
    print(f"✅ Applied DoRA to AST model with r={r}, alpha={alpha}")
    print(f"Target modules: {target_modules}")
    
    return ast_model_with_lora


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based audio processing.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MelSpectrogramConverter(nn.Module):
    """
    Convert raw audio to mel spectrogram using learnable parameters.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        # Create mel filter bank manually
        n_stft = n_fft // 2 + 1
        mel_filters = self._create_mel_filter_bank(n_mels, n_stft, f_min, self.f_max, sample_rate)
        self.register_buffer('mel_filters', mel_filters)
        
        # Create window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
    
    def _create_mel_filter_bank(self, n_mels, n_stft, f_min, f_max, sample_rate):
        """Create mel filter bank manually."""
        # Convert frequencies to mel scale
        def hz_to_mel(freq):
            return 2595 * torch.log10(1 + freq / 700.0)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel scale points
        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to bin indices
        bin_points = (hz_points / sample_rate * n_stft).long()
        
        # Create filter bank
        filter_bank = torch.zeros(n_mels, n_stft)
        
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising edge
            for j in range(left, center):
                if j < n_stft:
                    filter_bank[i, j] = (j - left) / (center - left)
            
            # Falling edge
            for j in range(center, right):
                if j < n_stft:
                    filter_bank[i, j] = (right - j) / (right - center)
        
        return filter_bank
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio to mel spectrogram.
        
        Args:
            audio: Raw audio tensor of shape [batch_size, seq_length]
        
        Returns:
            mel_spectrogram: Mel spectrogram of shape [batch_size, n_mels, time_frames]
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        # Convert to magnitude
        magnitude = torch.abs(stft)
        
        # Apply mel filter bank
        mel_spectrogram = torch.matmul(self.mel_filters, magnitude)
        
        # Convert to log scale
        mel_spectrogram = torch.log(mel_spectrogram + 1e-8)
        
        return mel_spectrogram


class ASTEncoder(EncoderCore):
    """
    Audio Spectrogram Transformer encoder using transformers library's pretrained AST model.
    
    This encoder uses the pretrained AST model from transformers library for better performance
    and leverages pre-trained weights from AudioSet or other audio datasets.
    """
    
    def __init__(
        self,
        input_shape,              # Required by EncoderCore
        sample_rate: int = 16000, # Audio sample rate
        n_fft: int = 1024,        # FFT window size (used for feature extractor)
        hop_length: int = 512,    # Hop length for STFT (used for feature extractor)
        n_mels: int = 128,        # Number of mel bins (used for feature extractor)
        d_model: int = 768,       # Model dimension (AST base model uses 768)
        nhead: int = 12,          # Number of attention heads (AST base model uses 12)
        num_layers: int = 12,     # Number of transformer layers (AST base model uses 12)
        dim_feedforward: int = 3072,  # Feedforward dimension (AST base model uses 3072)
        dropout: float = 0.1,     # Dropout rate
        patch_size: int = 16,     # Patch size for spectrogram (AST uses 16x16)
        max_audio_length: float = 10.0,  # Maximum expected audio length
        use_pretrained: bool = True,  # Whether to use pretrained weights
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",  # Pretrained model name
        # DoRA configuration
        use_dora: bool = False,   # Whether to use DoRA for finetuning
        dora_r: int = 8,          # DoRA rank
        dora_alpha: int = 16,     # DoRA alpha
        dora_target_modules: list = None,  # DoRA target modules
        dora_dropout: float = 0.1,  # DoRA dropout
    ):
        super().__init__(input_shape)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for ASTEncoder. Please install with: pip install transformers")
        
        self.sample_rate = sample_rate
        self.d_model = d_model
        self.use_pretrained = use_pretrained
        self.pretrained_name = pretrained_name
        
        # DoRA configuration
        self.use_dora = use_dora
        self.dora_r = dora_r
        self.dora_alpha = dora_alpha
        self.dora_target_modules = dora_target_modules or [
            "query", "key", "value", "output.dense", 
            "intermediate.dense", "attention.output.dense"
        ]
        self.dora_dropout = dora_dropout
        
        if self.use_pretrained:
            # Load pretrained AST model and feature extractor
            print(f"Loading pretrained AST model: {pretrained_name}")
            self.ast = ASTModel.from_pretrained(pretrained_name)
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_name)
            
            # Get the actual model dimension from pretrained model
            self.d_model = self.ast.config.hidden_size
            
            # Apply DoRA if enabled
            if self.use_dora:
                if not PEFT_AVAILABLE:
                    raise ImportError("PEFT library is required for DoRA. Please install with: pip install peft")
                
                print(f"🔧 Applying DoRA to AST model with r={self.dora_r}, alpha={self.dora_alpha}")
                self.ast = apply_dora_to_ast(
                    self.ast, 
                    target_modules=self.dora_target_modules,
                    r=self.dora_r, 
                    alpha=self.dora_alpha, 
                    dropout=self.dora_dropout
                )
                print("✅ DoRA applied successfully")
            else:
                # Freeze pretrained parameters when not using DoRA
                for param in self.ast.parameters():
                    param.requires_grad = False
                print("🔒 AST pretrained weights frozen - no gradient updates")
            
            print(f"✅ Loaded pretrained AST model with hidden_size: {self.d_model}")
            
            # Initialize projection layer from 768 to 512 dimensions
            self.to_512 = nn.Linear(768, 512)
        else:
            # Fallback to custom implementation if pretrained is not used
            print("Using custom AST implementation (not recommended)")
            self._init_custom_ast(
                sample_rate, n_fft, hop_length, n_mels, d_model, 
                nhead, num_layers, dim_feedforward, dropout, patch_size, max_audio_length
            )
    
    def _init_custom_ast(self, sample_rate, n_fft, hop_length, n_mels, d_model, 
                        nhead, num_layers, dim_feedforward, dropout, patch_size, max_audio_length):
        """Initialize custom AST implementation as fallback."""
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.max_audio_length = max_audio_length
        
        # Store original d_model for output projection
        original_d_model = d_model
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            # Adjust d_model to be divisible by nhead
            d_model = ((d_model + nhead - 1) // nhead) * nhead
            print(f"Warning: Adjusted d_model to {d_model} to be divisible by nhead={nhead}")
        
        # Update self.d_model to the adjusted value
        self.d_model = d_model
        
        # Calculate mel spectrogram dimensions (will be computed dynamically)
        self.mel_height = n_mels
        self.num_patches_height = self.mel_height // patch_size
        
        # Mel spectrogram converter
        self.mel_converter = MelSpectrogramConverter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Patch embedding: convert patches to embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=1,  # Single channel mel spectrogram
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional encoding with maximum expected length
        max_audio_samples = int(sample_rate * max_audio_length)
        max_mel_width = int((max_audio_samples - n_fft) // hop_length) + 1
        max_num_patches_width = max_mel_width // patch_size
        max_num_patches = self.num_patches_height * max_num_patches_width
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_num_patches)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_random_weights()
        
        # Initialize output projection after d_model is set
        self.output_proj = nn.Linear(self.d_model, original_d_model)
    
    def _init_random_weights(self):
        """Initialize with random weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AST encoder.
        
        Args:
            x: Input raw audio of shape [batch_size, audio_samples]
               Supports variable length audio input
        
        Returns:
            Encoded features of shape [batch_size, d_model]
        """
        if self.use_pretrained:
            return self._forward_pretrained(x)
        else:
            return self._forward_custom(x)
    
    def _forward_pretrained(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pretrained AST model."""
        batch_size = x.size(0)
        
        # Convert tensor to numpy for feature extractor
        # ASTFeatureExtractor expects numpy arrays
        audio_arrays = []
        for i in range(batch_size):
            audio_array = x[i].detach().cpu().numpy()
            audio_arrays.append(audio_array)
        
        # Extract features using ASTFeatureExtractor
        # This handles mel spectrogram conversion and normalization
        inputs = self.feature_extractor(
            audio_arrays, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        device = next(self.ast.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Forward pass through pretrained AST model
        if self.use_dora:
            # With DoRA, we need gradients for training
            outputs = self.ast(**inputs)
        else:
            # Without DoRA, use no_grad since weights are frozen
            with torch.no_grad():
                outputs = self.ast(**inputs)
        
        # Get the pooled output (CLS token representation)
        # AST model outputs last_hidden_state of shape [batch_size, seq_len, hidden_size]
        # We use the first token (CLS token) as the global representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Project from 768 to 512 dimensions
        output = self.to_512(pooled_output)  # [batch_size, 512]
        
        # # Debug: Print final output information
        # if hasattr(self, 'use_dora') and self.use_dora:
        #     print(f"🎵 DoRA AST CLS token shape: {pooled_output.shape}")
        #     print(f"🎵 DoRA AST CLS token norm: {torch.norm(pooled_output).item():.4f}")
        #     print(f"🎵 DoRA AST Final output shape: {output.shape}")
        #     print(f"🎵 DoRA AST Final output norm: {torch.norm(output).item():.4f}")
        #     print("=" * 50)
        
        return output
    
    def _forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom AST implementation."""
        batch_size = x.size(0)
        audio_samples = x.size(1)
        
        # Convert raw audio to mel spectrogram
        mel_spectrogram = self.mel_converter(x)
        
        # Add channel dimension for conv2d: [batch_size, n_mels, time_frames] -> [batch_size, 1, n_mels, time_frames]
        mel_spectrogram = mel_spectrogram.unsqueeze(1)
        
        # Patch embedding: [batch_size, 1, height, width] -> [batch_size, d_model, num_patches_h, num_patches_w]
        x = self.patch_embedding(mel_spectrogram)
        
        # Reshape to sequence: [batch_size, d_model, num_patches_h, num_patches_w] -> [batch_size, num_patches, d_model]
        x = x.flatten(2).transpose(1, 2)
        
        # Get actual number of patches for this input
        num_patches = x.size(1)
        
        # Add positional encoding (only use the required length)
        x = x.transpose(0, 1)  # [num_patches, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, num_patches, d_model]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Apply layer normalization and output projection
        x = self.layer_norm(x)
        x = self.output_proj(x)
        
        return x
    
    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output shape of the encoder.
        
        Args:
            input_shape: Input shape tuple (should be [audio_samples])
        
        Returns:
            Output shape tuple [512] (projected from 768)
        """
        return (512,)


class SharedASTEncoder(nn.Module):
    """
    Shared AST encoder for both robot and human data.
    This ensures consistent audio feature extraction across modalities.
    """
    
    def __init__(self, input_shape=(64000,), **kwargs):
        super().__init__()
        self.encoder = ASTEncoder(input_shape=input_shape, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared encoder.
        
        Args:
            x: Input raw audio
        
        Returns:
            Encoded audio features
        """
        return self.encoder(x)
    
    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return output shape of the encoder."""
        return self.encoder.output_shape(input_shape)


def create_ast_encoder(
    input_shape=(64000,),
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    d_model: int = 512,  # Output dimension after projection from 768
    max_audio_length: float = 10.0,
    use_pretrained: bool = True,
    pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    # DoRA configuration
    use_dora: bool = False,
    dora_r: int = 8,
    dora_alpha: int = 16,
    dora_target_modules: list = None,
    dora_dropout: float = 0.1,
    **kwargs
) -> SharedASTEncoder:
    """
    Factory function to create a shared AST encoder.
    
    Args:
        input_shape: Input audio shape (default: (64000,) for 4 seconds at 16kHz)
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
        d_model: Model dimension (768 for AST base model)
        max_audio_length: Maximum expected audio length for positional encoding
        use_pretrained: Whether to use pretrained AST model
        pretrained_name: Name of pretrained model to load
        use_dora: Whether to use DoRA for efficient finetuning
        dora_r: DoRA rank parameter
        dora_alpha: DoRA alpha parameter
        dora_target_modules: List of module names to apply DoRA to
        dora_dropout: DoRA dropout rate
        **kwargs: Additional arguments for ASTEncoder
    
    Returns:
        SharedASTEncoder instance
    """
    return SharedASTEncoder(
        input_shape=input_shape,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        d_model=d_model,
        max_audio_length=max_audio_length,
        use_pretrained=use_pretrained,
        pretrained_name=pretrained_name,
        use_dora=use_dora,
        dora_r=dora_r,
        dora_alpha=dora_alpha,
        dora_target_modules=dora_target_modules,
        dora_dropout=dora_dropout,
        **kwargs
    )


class AudioModality(Modality):
    """
    Modality for raw audio observations
    """
    name = "audio"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Process raw audio for network input.
        Converts to float tensor and ensures proper shape.
        
        Args:
            obs (np.array or torch.Tensor): raw audio array
            
        Returns:
            processed_obs (torch.Tensor): processed raw audio
        """
        if isinstance(obs, torch.Tensor):
            return obs.float()
        else:
            return torch.tensor(obs, dtype=torch.float32)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Prepare audio data for saving to dataset.
        
        Args:
            obs (torch.Tensor): processed raw audio
            
        Returns:
            unprocessed_obs (torch.Tensor): raw audio ready for saving
        """
        return obs


def get_audio_feature_dim():
    """
    Get the output feature dimension of the AST encoder.
    
    Returns:
        int: Feature dimension (512 after projection from 768)
    """
    return 512  # Projected from AST base model hidden size (768)
