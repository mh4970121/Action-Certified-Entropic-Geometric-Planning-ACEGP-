import torch
import numpy as np

class SplitConformalCertifier:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, nonconformity_scores: np.ndarray):
        n = len(nonconformity_scores)
        q_val = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(nonconformity_scores, q_val, method="higher")
        return self.q_hat

    def predict_bound(self, predicted_width: torch.Tensor) -> torch.Tensor:
        assert self.q_hat is not None, "Conformal certifier must be calibrated first."
        return predicted_width * self.q_hat
