"""
Velocity Estimator Module for PIP-Loco
Standalone TCN-based network for estimating robot body velocity from proprioceptive history.
"""

import torch
import torch.nn as nn
from typing import List


class VelocityEstimator(nn.Module):
    """
    Temporal Convolutional Network (TCN) for estimating robot body linear velocity.
    
    Architecture:
        - TCN Backbone: 3x Conv1d layers (48 -> 128 -> 64 -> 32) with ReLU + BatchNorm
        - MLP Head: Flatten -> Linear(flatten_dim, 128) -> ReLU -> Linear(128, 3)
    
    Input:
        x: torch.Tensor of shape (Batch, History_Length, Input_Dim)
           - Batch: variable batch size
           - History_Length: temporal window (default 50 steps = 1.0s @ 50Hz)
           - Input_Dim: proprioceptive features (default 48 channels)
    
    Output:
        velocity: torch.Tensor of shape (Batch, 3)
                  Estimated body linear velocity [v_x, v_y, v_z]
    """
    
    def __init__(
        self,
        input_dim: int = 48,
        history_length: int = 50,
        hidden_dims: List[int] = None,
        output_dim: int = 3
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.history_length = history_length
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build TCN backbone
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in hidden_dims:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ])
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*conv_layers)
        
        # Dynamically calculate flattened dimension
        flatten_dim = hidden_dims[-1] * history_length
        
        # Build MLP head
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the velocity estimator.
        
        Args:
            x: Input tensor of shape (Batch, History_Length, Input_Dim)
        
        Returns:
            Estimated velocity tensor of shape (Batch, 3)
        """
        # Transpose: (Batch, Length, Channels) -> (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        
        # TCN feature extraction
        x = self.tcn(x)
        
        # Flatten: (Batch, Channels, Length) -> (Batch, Channels * Length)
        x = x.flatten(start_dim=1)
        
        # MLP regression head
        velocity = self.mlp(x)
        
        return velocity


