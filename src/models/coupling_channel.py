import torch
import torch.nn as nn

class RankRProjectionChannel(nn.Module):
    def __init__(self, latent_dim: int = 128, rank: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        
        # Define top-r subspace projection matrix \Pi = \sum_{i=1}^r e_i e_i^\top
        pi = torch.zeros(latent_dim, latent_dim)
        if rank > 0:
            pi[:rank, :rank] = torch.eye(rank)
        self.register_buffer("Pi", pi)
        
    def forward(self, z_grad_flow: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            # Completely decoupled: Stop-gradient on full vector
            return z_grad_flow.detach()
        elif self.rank == self.latent_dim:
            # Fully coupled: Full gradient pass-through
            return z_grad_flow
        else:
            # Confined Coupling: Route gradient through top-r subspace, freeze complement
            z_coupled = z_grad_flow @ self.Pi
            z_decoupled = (z_grad_flow @ (torch.eye(self.latent_dim, device=z_grad_flow.device) - self.Pi)).detach()
            return z_coupled + z_decoupled
