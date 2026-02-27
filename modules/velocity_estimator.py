"""
Velocity Estimator for PIP-Loco
Predicts body velocity from a history of blind observations
Input: (Batch , History_Length , Input_Dim). Output: (Batch , 3).
"""

import torch
import torch.nn as nn
from typing import List


class VelocityEstimator(nn.Module):
    """
    Temporal Convolutional Network (TCN) for Body Velocity Estimation
    Takes in a history of motion data to predict body velocity (v_x, v_y, v_z)
    Encoder: Three 1D convolutional layers (48 -> 128 -> 64 -> 32) with ReLU and BatchNorm.
    Head: Flattens temporal features into a two-layer MLP (Linear 128 -> Linear 3).
    """
    
    def __init__(
        self,
        input_dim: int = 45,
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
        
        # TCN : Conv -> ReLU -> BN blocks
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
        
        # Flatten dimension: channels * temporal length
        flatten_dim = hidden_dims[-1] * history_length
        
        # MLP head for regression
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the velocity estimator
        """
        # Swap dims for Conv1d: (B,L,C) -> (B,C,L)
        obs_history = obs_history.permute(0, 2, 1)
        x = self.tcn(obs_history)
        
        # Flatten temporal features for the MLP
        x = x.flatten(start_dim=1)
        velocity = self.mlp(x)
        
        return velocity


