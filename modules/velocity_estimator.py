"""
Velocity Estimator for PIP-Loco.
Standalone TCN regressor for body linear velocity.
Input: (Batch, History_Length, Input_Dim). Output: (Batch, 3).
Dims swapped to (B, C, L); features flattened then mapped by MLP.
"""

import torch
import torch.nn as nn
from typing import List


class VelocityEstimator(nn.Module):
    """
    TCN for estimating body linear velocity from proprioceptive history.
    
    Backbone: 3×Conv1d with ReLU+BatchNorm (48→128→64→32), stride=1, padding=1
    preserves temporal length. Head: Flatten (last_channels×history_length)
    → Linear(128) → ReLU → Linear(3).
    
    Input: torch.Tensor (Batch, History_Length, Input_Dim).
    Output: torch.Tensor (Batch, 3) as [v_x, v_y, v_z].
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
        
        # TCN backbone: Conv → ReLU → BN blocks, time preserved
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
        
        # Flatten dimension: channels × temporal length
        flatten_dim = hidden_dims[-1] * history_length
        
        # MLP head for regression
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
        # Swap dims for Conv1d: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)
        
        x = self.tcn(x)
        
        # Flatten temporal features for the MLP
        x = x.flatten(start_dim=1)
        
        velocity = self.mlp(x)
        
        return velocity


