"""
Standalone PyLaia-compatible CRNN model for use as readability supervisor in DiffBrush.
This recreates the PyLaia LaiaCRNN architecture to load pretrained checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block matching PyLaia's implementation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activation: str = 'LeakyReLU',
        use_batchnorm: bool = True,
        poolsize: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2 * dilation
        
        # Use a module dict to match PyLaia's naming convention
        self.block = nn.ModuleDict()
        
        self.block['conv'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not use_batchnorm
        )
        
        if use_batchnorm:
            self.block['batchnorm'] = nn.BatchNorm2d(out_channels)
        
        if activation == 'LeakyReLU':
            self.block['activation'] = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'ReLU':
            self.block['activation'] = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.block['activation'] = nn.Tanh()
        
        if poolsize is not None:
            self.block['pool'] = nn.MaxPool2d(kernel_size=poolsize, stride=poolsize)
        
        if dropout > 0:
            self.block['dropout'] = nn.Dropout2d(p=dropout)
        
        self.use_batchnorm = use_batchnorm
        self.has_pool = poolsize is not None
        self.has_dropout = dropout > 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block['conv'](x)
        if self.use_batchnorm:
            x = self.block['batchnorm'](x)
        x = self.block['activation'](x)
        if self.has_pool:
            x = self.block['pool'](x)
        if self.has_dropout:
            x = self.block['dropout'](x)
        return x


class PyLaiaCRNN(nn.Module):
    """
    CRNN model matching PyLaia's LaiaCRNN architecture.
    
    Architecture from model_v22 config:
    - CNN: 4 layers with features [32, 64, 128, 256]
    - Pooling: [[2,2], [2,2], [2,1], [2,1]] -> reduces height by 16, width by 4
    - RNN: 3-layer bidirectional LSTM with 512 units
    - Output: Linear projection to num_classes
    """
    
    def __init__(
        self,
        num_input_channels: int = 1,
        num_output_labels: int = 60,
        cnn_num_features: List[int] = [32, 64, 128, 256],
        cnn_kernel_size: List[int] = [3, 3, 3, 3],
        cnn_stride: List[int] = [1, 1, 1, 1],
        cnn_dilation: List[int] = [1, 1, 1, 1],
        cnn_activation: List[str] = ['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'],
        cnn_batchnorm: List[bool] = [True, True, True, True],
        cnn_poolsize: List[List[int]] = [[2, 2], [2, 2], [2, 1], [2, 1]],
        cnn_dropout: List[float] = [0.0, 0.0, 0.2, 0.2],
        rnn_type: str = 'LSTM',
        rnn_layers: int = 3,
        rnn_units: int = 512,
        rnn_dropout: float = 0.5,
        lin_dropout: float = 0.5,
        fixed_input_height: int = 128,
        adaptive_pooling: str = 'avg'
    ):
        super().__init__()
        
        self.fixed_input_height = fixed_input_height
        self.adaptive_pooling = adaptive_pooling
        self.num_output_labels = num_output_labels
        
        # Build CNN layers
        cnn_layers = []
        in_channels = num_input_channels
        
        for i, out_channels in enumerate(cnn_num_features):
            poolsize = tuple(cnn_poolsize[i]) if cnn_poolsize[i] else None
            cnn_layers.append(ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=cnn_kernel_size[i],
                stride=cnn_stride[i],
                dilation=cnn_dilation[i],
                activation=cnn_activation[i],
                use_batchnorm=cnn_batchnorm[i],
                poolsize=poolsize,
                dropout=cnn_dropout[i]
            ))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate remaining height after all pooling
        # Height: 128 -> 64 -> 32 -> 16 -> 8 (with poolsizes [[2,2], [2,2], [2,1], [2,1]])
        remaining_height = fixed_input_height
        for pool in cnn_poolsize:
            if pool is not None:
                remaining_height = remaining_height // pool[0]
        
        self.remaining_height = remaining_height
        
        # Adaptive pooling to optionally collapse height
        # PyLaia uses adaptive pooling but the RNN input includes the height dimension
        if adaptive_pooling == 'avg':
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
            rnn_input_size = cnn_num_features[-1]  # Height collapsed to 1
        elif adaptive_pooling == 'max':
            self.adaptive_pool = nn.AdaptiveMaxPool2d((1, None))
            rnn_input_size = cnn_num_features[-1]  # Height collapsed to 1
        else:
            self.adaptive_pool = None
            # Height not collapsed, flattened with channels
            rnn_input_size = cnn_num_features[-1] * remaining_height
        
        # Override: PyLaia actually flattens height with channels (256 * 8 = 2048)
        # even when adaptive_pooling is specified, it seems to work differently
        # Based on checkpoint, RNN input is 2048 = 256 channels * 8 height
        self.adaptive_pool = None  # Don't use adaptive pooling
        rnn_input_size = cnn_num_features[-1] * remaining_height  # 256 * 8 = 2048
        
        # Build RNN
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_units,
                num_layers=rnn_layers,
                dropout=rnn_dropout if rnn_layers > 1 else 0,
                bidirectional=True,
                batch_first=False
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_units,
                num_layers=rnn_layers,
                dropout=rnn_dropout if rnn_layers > 1 else 0,
                bidirectional=True,
                batch_first=False
            )
        
        # Linear dropout
        self.lin_dropout = nn.Dropout(p=lin_dropout)
        
        # Output projection (bidirectional -> 2 * rnn_units)
        self.output = nn.Linear(2 * rnn_units, num_output_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W] where H=128 (fixed height)
        
        Returns:
            Log probabilities of shape [T, B, num_classes] for CTC
        """
        # CNN feature extraction
        # [B, C, H, W] -> [B, 256, 8, W/4]
        x = self.cnn(x)
        
        # Get dimensions
        batch_size, channels, height, width = x.shape
        
        if self.adaptive_pool is not None:
            # Adaptive pooling to collapse height
            # [B, 256, H', W'] -> [B, 256, 1, W']
            x = self.adaptive_pool(x)
            x = x.squeeze(2)  # [B, 256, W']
        else:
            # Flatten height with channels: [B, 256, 8, W] -> [B, 2048, W]
            x = x.permute(0, 3, 1, 2)  # [B, W, 256, 8]
            x = x.reshape(batch_size, width, channels * height)  # [B, W, 2048]
            x = x.permute(0, 2, 1)  # [B, 2048, W]
        
        # Prepare for RNN: [B, C, W] -> [W, B, C]
        x = x.permute(2, 0, 1)  # [W, B, C] for RNN
        
        # RNN
        # [T, B, 2048] -> [T, B, 1024] (bidirectional with 512 units)
        x, _ = self.rnn(x)
        
        # Dropout + Linear
        x = self.lin_dropout(x)
        x = self.output(x)
        
        # Log softmax for CTC
        x = F.log_softmax(x, dim=-1)
        
        return x


def load_pylaia_checkpoint(model: PyLaiaCRNN, checkpoint_path: str, device: torch.device) -> PyLaiaCRNN:
    """
    Load a PyLaia checkpoint into the model.
    
    Supports two formats:
    1. Weights-only (.pth) - just a state_dict, no Lightning dependencies
    2. Lightning checkpoint (.ckpt) - requires pytorch_lightning installed
    
    For Lightning checkpoints without pytorch_lightning installed, first extract
    the weights using the pylaia environment:
        python -c "
        import torch
        ckpt = torch.load('checkpoint.ckpt', map_location='cpu')
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        torch.save(state_dict, 'weights_only.pth')
        "
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    else:
        # Weights-only format (already just a state_dict)
        state_dict = checkpoint
    
    # Map PyLaia layer names to our model
    # PyLaia uses: conv.0.0, conv.0.1, conv.1.0, ... 
    # Our model uses: cnn.0.block.0, cnn.0.block.1, ...
    mapped_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Map CNN layers: conv.X.Y -> cnn.X.block.Y
        if key.startswith('conv.'):
            parts = key.split('.')
            layer_idx = int(parts[1])
            rest = '.'.join(parts[2:])
            new_key = f'cnn.{layer_idx}.block.{rest}'
        
        # Map RNN layers (should match directly)
        # Map output layer: linear -> output
        elif key.startswith('linear.'):
            new_key = key.replace('linear.', 'output.')
        
        mapped_state_dict[new_key] = value
    
    # Try to load with mapped keys
    try:
        model.load_state_dict(mapped_state_dict, strict=True)
        print(f"✓ Loaded PyLaia checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying non-strict: {e}")
        # If mapping failed, try loading with original keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")
    
    return model


class SymbolTable:
    """Symbol table for encoding/decoding text for CTC loss."""
    
    def __init__(self, syms_path: str):
        """
        Load symbol table from PyLaia syms.txt file.
        Format: symbol index
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.blank_idx = 0  # CTC blank is usually 0
        
        with open(syms_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.rsplit(' ', 1)  # Split from right to handle space character
                    if len(parts) == 2:
                        char, idx = parts
                        idx = int(idx)
                        self.char_to_idx[char] = idx
                        self.idx_to_char[idx] = char
        
        # Handle special tokens
        self.space_idx = self.char_to_idx.get('<space>', 1)
        self.unk_idx = self.char_to_idx.get('<unk>', 2)
        
        print(f"Loaded symbol table with {len(self.char_to_idx)} symbols")
    
    def encode(self, text: str) -> List[int]:
        """Encode text string to list of indices."""
        indices = []
        for char in text:
            if char == ' ':
                indices.append(self.space_idx)
            elif char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.unk_idx)
        return indices
    
    def decode(self, indices: List[int], remove_blanks: bool = True, collapse_repeats: bool = True) -> str:
        """Decode list of indices to text string."""
        chars = []
        prev_idx = None
        
        for idx in indices:
            if remove_blanks and idx == self.blank_idx:
                prev_idx = idx
                continue
            if collapse_repeats and idx == prev_idx:
                continue
            
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char == '<space>':
                    chars.append(' ')
                elif char not in ['<ctc>', '<unk>']:
                    chars.append(char)
            prev_idx = idx
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char_to_idx)


def create_pylaia_supervisor(
    checkpoint_path: str,
    syms_path: str,
    device: torch.device
) -> Tuple[PyLaiaCRNN, SymbolTable]:
    """
    Create and load a frozen PyLaia model for use as readability supervisor.
    
    Args:
        checkpoint_path: Path to PyLaia .ckpt file
        syms_path: Path to syms.txt file
        device: Device to load model on
    
    Returns:
        Tuple of (frozen model, symbol table)
    """
    # Load symbol table
    symbol_table = SymbolTable(syms_path)
    
    # Create model with Latin BHO config
    model = PyLaiaCRNN(
        num_input_channels=1,
        num_output_labels=len(symbol_table),
        cnn_num_features=[32, 64, 128, 256],
        cnn_kernel_size=[3, 3, 3, 3],
        cnn_stride=[1, 1, 1, 1],
        cnn_dilation=[1, 1, 1, 1],
        cnn_activation=['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'],
        cnn_batchnorm=[True, True, True, True],
        cnn_poolsize=[[2, 2], [2, 2], [2, 1], [2, 1]],
        cnn_dropout=[0.0, 0.0, 0.2, 0.2],
        rnn_type='LSTM',
        rnn_layers=3,
        rnn_units=512,
        rnn_dropout=0.5,
        lin_dropout=0.5,
        fixed_input_height=128,
        adaptive_pooling='avg'
    )
    
    # Load checkpoint
    model = load_pylaia_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    
    # Freeze model parameters (but keep in training mode for cuDNN RNN backward compatibility)
    # cuDNN RNN backward requires training mode, so we can't use eval()
    # Instead, we freeze parameters and disable dropout manually
    for param in model.parameters():
        param.requires_grad = False
    
    # Keep model in training mode (required for cuDNN RNN backward)
    # But disable dropout by setting p=0
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.p = 0.0
        # BatchNorm will use running stats in eval mode, but we need train mode
        # So we freeze the running stats manually
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # Use running mean/var
    
    # Note: model.train() keeps RNN in training mode (needed for cuDNN backward)
    # but parameters are frozen so nothing will be updated
    model.train()
    
    print(f"✓ PyLaia supervisor ready (frozen, {sum(p.numel() for p in model.parameters())} params)")
    
    return model, symbol_table

