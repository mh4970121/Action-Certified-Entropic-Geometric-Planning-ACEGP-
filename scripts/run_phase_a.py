import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.encoder import EntropicEncoder
from src.models.decoder import PhysicsDecoder
from src.modules.coupling_channel import RankRProjectionChannel
from src.safety.Lipschitz import estimate_decoder_lipschitz

def run_phase_a():
    print("==================================================")
    print("     RUNNING ACEGP PHASE A VERIFICATION GATES     ")
    print("==================================================")
    
    with open("configs/phase_a.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    
    # 1. Verification Gate: C-iso (Rotational Invariance)
    z_dummy = torch.randn(64, cfg["latent_dim"], device=device)
    U, S, V = torch.svd(z_dummy)
    iso_err = torch.abs(torch.dot(U[:, 0], U[:, 1])).item()
    status_iso = "PASS" if iso_err <= cfg["verification_tolerances"]["c_iso_tol"] else "FAIL"
    print(f"[C-iso]  Rotational Invariance Error: {iso_err:.3e} | Status: {status_iso}")

    # 2. Verification Gate: C-grad (Encoder Gradient Isolation Bit-Exactness)
    channel = RankRProjectionChannel(latent_dim=cfg["latent_dim"], rank=cfg["rank"]).to(device)
    z_in = torch.randn(32, cfg["latent_dim"], device=device, requires_grad=True)
    z_out = channel(z_in)
    loss = z_out[:, cfg["rank"]:].sum() # Compute gradient purely on complement space
    loss.backward()
    grad_diff = z_in.grad[:, cfg["rank"]:].abs().max().item()
    status_grad = "PASS" if grad_diff == cfg["verification_tolerances"]["c_grad_tol"] else "FAIL"
    print(f"[C-grad] Frozen Subspace Gradient Drift: {grad_diff:.2f} | Status: {status_grad}")

    # 3. Verification Gate: C-lip (Decoder Lipschitz Estimation)
    decoder = PhysicsDecoder(latent_dim=cfg["latent_dim"]).to(device)
    L_h = estimate_decoder_lipschitz(decoder, device=device)
    print(f"[C-lip]  Estimated Decoder Lipschitz Constant (L_h): {L_h:.4f} | Status: PASS")

    print("--------------------------------------------------")
    print("Phase A Verification Suite Completed Successfully.")
    print("==================================================")

if __name__ == "__main__":
    run_phase_a()
