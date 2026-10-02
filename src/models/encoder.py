import torch
import torch.nn as nn

class EntropicEncoder(nn.Module):
    def __init__(self, obs_dim: int = 5, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # Apply variance-covariance / entropic standardization constraint
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, keepdim=True) + 1e-5)
        return z / std
