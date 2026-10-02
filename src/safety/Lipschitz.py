import torch
import torch.nn as nn

def estimate_decoder_lipschitz(decoder: nn.Module, num_iterations: int = 100, device: str = "cuda:0") -> float:
    """Estimates decoder Lipschitz constant L_h via power iteration on linear weights."""
    l_h = 1.0
    for name, module in decoder.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            u = torch.randn(weight.size(1), 1, device=device)
            u = u / torch.norm(u)
            for _ in range(num_iterations):
                v = weight @ u
                v = v / torch.norm(v)
                u = weight.T @ v
                u = u / torch.norm(u)
            sigma = torch.norm(weight @ u).item()
            l_h *= sigma
    return l_h
