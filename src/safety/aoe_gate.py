import torch

class AbstainOrExecuteGate:
    def __init__(self, tau_task: float = 1.0):
        self.tau_task = tau_task

    def evaluate(self, certified_bound: float, g_health_pass: bool) -> str:
        """Runtime Abstain-or-Execute (AoE) decision contract."""
        if not g_health_pass:
            return "ABSTAIN_HEALTH_FAIL"
        if certified_bound > self.tau_task:
            return "ABSTAIN_BOUND_EXCEEDED"
        return "EXECUTE"
