import torch
import torch.nn as nn

class CertificateHead(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, q: float = 0.90) -> torch.Tensor:
    errors = y_true - y_pred
    return torch.max((q - 1) * errors, q * errors).mean()
