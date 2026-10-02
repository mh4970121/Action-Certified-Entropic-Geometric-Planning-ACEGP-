import torch
import torch.nn as nn

class VHTGuard(nn.Module):
    def __init__(self, lambda_vht: float = 0.1):
        super().__init__()
        self.lambda_vht = lambda_vht

    def compute_guard_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        # Penalizes unconditioned subspace drift (Verified Hindsight Training)
        residual = z_pred - z_target
        return self.lambda_vht * torch.norm(residual, p=2, dim=-1).mean()
