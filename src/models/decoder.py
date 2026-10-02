import torch
import torch.nn as nn

class PhysicsDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, pose_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
