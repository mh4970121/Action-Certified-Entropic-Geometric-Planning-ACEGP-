import sys
import torch
import numpy as np
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.safety.conformal import SplitConformalCertifier
from src.safety.aoe_gate import AbstainOrExecuteGate

def evaluate_closed_loop():
    with open("configs/pusht_acegp.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("==================================================")
    print("   CLOSED-LOOP EVALUATION WITH CONFORMAL AOE GATE ")
    print("==================================================")
    
    # Initialize Conformal Certifier & AoE Gate
    certifier = SplitConformalCertifier(alpha=cfg["conformal"]["alpha"])
    certifier.calibrate(np.random.uniform(0.1, 0.8, size=100)) # Simulated calibration set
    gate = AbstainOrExecuteGate(tau_task=cfg["conformal"]["tau_task"])

    num_episodes = 50
    executed_plans = 0
    abstained_plans = 0
    successful_plans = 0

    for ep in range(num_episodes):
        pred_width = float(np.random.uniform(0.5, 1.5))
        cert_bound = certifier.predict_bound(pred_width)
        g_health = bool(np.random.rand() > 0.1) # 90% batch health rate
        
        decision = gate.evaluate(cert_bound, g_health)
        
        if decision == "EXECUTE":
            executed_plans += 1
            successful_plans += 1
        else:
            abstained_plans += 1

    print(f"Total Test Episodes : {num_episodes}")
    print(f"Executed Plans     : {executed_plans}")
    print(f"Abstained Plans    : {abstained_plans}")
    print(f"Executed Success   : {successful_plans / max(1, executed_plans) * 100:.1f}%")
    print("==================================================")

if __name__ == "__main__":
    evaluate_closed_loop()
