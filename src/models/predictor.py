import torch
import torch.nn as nn

class HybridPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)
        return self.net(x)
