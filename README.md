# Learning Fidelity Susceptibility from Quantum-Simulator Snapshots

This repository supports a mini-project on learning Born distributions from quantum-simulator snapshots and using a classical proxy for the fidelity susceptibility to locate a phase transition in the 1D spinless t--V model.

The project handout with full task details is in `Notes/main.pdf`.

## Repository layout
- `Mini_Project/rnn_model.py`: Conditional RNN autoregressive model (JAX/Flax) with sampling and likelihood utilities.
- `Mini_Project/generate_samples.py`: Generates snapshot datasets from exact diagonalization using QuSpin.
- `Mini_Project/analyze_samples.py`: Example analysis script: load data, evaluate NLL, sample the RNN, compute an order parameter.
- `Notes/main.pdf`: Project statement, background, and checklist.
- `checkpoints/`: Place to store trained model checkpoints (create as needed).

## Setup
Recommended: create a fresh Python environment and install dependencies.

The key packages and their tested version are 
- `numpy==2.1.3`, `scipy==1.15.3`, `matplotlib`
- `jax==0.6.0`, `jaxlib==0.6.0`, `flax==0.10.4`
- `quspin==1.0.0`

Example (adapt to your environment):
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib jax flax quspin
```

If you use a GPU, install the matching JAX build from the official JAX docs.

## Quick start
1) Generate snapshot datasets (exact diagonalization) -- if you participate in the 'Quantum ideas factory challenge' workshop, the snapshot datasets will be provided for download.
```bash
python Mini_Project/generate_samples.py --sizes 10 12 14 --u-min 0.1 --u-max 6.0 --u-step 0.25 --nsamples 10000 --out-dir data
```

2) Run the example analysis (untrained RNN -- this is just a demo of I/O and metrics).
```bash
python Mini_Project/analyze_samples.py --data-dir data --Lx 12 --features 128 --num-samples 2048
```

## What you need to implement
Follow `Instructions.pdf` for the full requirements. In brief:
- Training routine for the conditional RNN on mixed-$\lambda$ batches (teacher-forced NLL).
- Routine to compute the Bhattacharyya-based fidelity susceptibility $\chi_{B(\lambda)}$.
- Routine to compute and plot the **charge density wave** (CDW) order parameter S(π) and its derivative.
- Plots:
  - Training/validation NLL vs epoch
  - $S(\pi)$ vs $\lambda$ and $dS/d\lambda$
  - χ_B(λ) vs λ showing a peak near the transition

## Tips
- Use periodic-translation data augmentation for PBCs (see checklist in `Instructions.pdf`).
- Keep the training/validation split stratified across $\lambda$ values.
- Validate NLL with teacher forcing even if you use scheduled sampling for training.

## Notes on data
The generator uses exact diagonalization and may take time for larger sizes or fine $\lambda$ grids. Start small (e.g., L=10 or L=12) and scale up if needed.