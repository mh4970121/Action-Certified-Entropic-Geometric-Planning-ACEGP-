# Action-Certified Entropic–Geometric Planning (ACEGP)
Confined Coupling and Conformal Action Certificates for Latent World Models


[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Pre-Registration](https://img.shields.io/badge/Pre--Registration-Verified-success.svg)](#pre-registration--falsification-protocol)

Official PyTorch implementation of **"Action-Certified Entropic–Geometric Planning (ACEGP): Confined Coupling and Conformal Action Certificates for Latent World Models"**.

---

## 📌 Abstract

 guarantees for latent world models have so far concerned *representations*: how latents are distributed and how well-conditioned the predictor is. Deployment requires guarantees about *actions*: whether the plan produced by the model will execute safely in physical space. 

**ACEGP** resolves this gap through two coordinated moves:
1. **Confined Coupling (Rank-$r$ Channel):** Transmits prediction loss into the encoder exclusively through an $r$-dimensional subspace $\Pi = \sum_{i=1}^r e_i e_i^\top$, leaving the $(K-r)$-dimensional complement covariance strictly frozen.
2. **Action Certification:** Combines split-conformal rollout certificates, a decoder-Lipschitz bridge to task space, and an **Abstain-or-Execute (AoE)** runtime contract backed by a batch-level health gate ($\Ghealth$).

On the frozen **Push-T** benchmark across 30 paired seeds:
* **Contact Selectivity:** Reaches **$3.42\times$** at the rule-selected rank $\hat r = 32$ ($p < 10^{-6}$), clearing the H1 threshold ($\ge 2.0\times$).
* **Closed-Loop Task Success:** Reaches **$64.0\%$** ($32/50$) under the conformal AoE gate (compared to $0/50$ for DEGP), with **$0\%$ task violations** among executed plans.

---

## 🏗️ System Architecture & Gradient Isolation

ACEGP maintains a strict **stop-gradient by default** policy across all model components, with exactly one explicit exception: the rank-$r$ projection channel $\Pi$.

# Clone repository
    git clone [https://github.com/mh4970121/Action-Certified-Entropic-Geometric-Planning-ACEGP-/tree/main.git](https://github.com/mh4970121/Action-Certified-Entropic-Geometric-Planning-ACEGP-/tree/main.git)
    cd ACEGP

# Create conda environment
conda create -n acegp python=3.10 -y
conda activate acegp

# Install dependencies
    pip install -r requirements.txt

1. Run Verification Protocol (Phase A)

python scripts/run_phase_a.py --config configs/phase_a.yaml

2. Train Model with Confined Coupling ($r=32$)

       python train.py \
          --config configs/pusht_acegp.yaml \
          --rank 32 \
          --seed 42 \
          --device cuda:0

3. Conformal Calibration (Stage B)

       python calibrate_conformal.py \
           --ckpt checkpoints/pusht_r32_seed42.pt \
           --mode B-PLAN \
           --alpha 0.10

4. Evaluate Closed-Loop Deployment with AoE Gate (Stage D)
  
       python evaluate_closed_loop.py \
           --ckpt checkpoints/pusht_r32_seed42.pt \
          --calib_file calibration_r32.pt \
          --aoe_gate \
          --tau_task 1.0 \
          --num_episodes 50   
