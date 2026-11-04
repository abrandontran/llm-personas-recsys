# Robust Reinforcement Learning for Recommender Systems via *In Silico* LLM Personas

[![Yale SDS 6190](https://img.shields.io/badge/Yale-SDS%206190-00356B.svg)](https://statistics.yale.edu/)
[![NeurIPS Preprint](https://img.shields.io/badge/Format-NeurIPS%202026-brightgreen.svg)](Proposal/project_proposal_v3.pdf)

Sequential decision making in recommendation systems via high-throughput LLM persona simulation, bridging the sim-to-real gap between parametric simulators and real-world offline logs.

## Repository Structure

```
├── artifacts/             # Precomputed embeddings, SVD representations & archetype JSONs
├── data/                  # Local datasets (MovieLens-1M, KuaiRec) [gitignored]
├── Midterm Presentation/  # Slide deck (Beamer .tex, .pdf, speaker notes, briefing)
├── Milestone 1/           # Intermediate progress report & evaluation benchmarks
├── Proposal/              # NeurIPS-format project proposal (v1, v2, v3)
├── scripts/               # End-to-end execution scripts (local & HPC)
├── slurm/                 # HPC Slurm batch scripts for Bouchet cluster
└── src/                   # Core Python package (envs, models, metrics, data)
```

## Setup & Workflow

- **Local Machine:** Data preprocessing, SVD item embeddings, archetype clustering, parametric RecSim ($\mathcal{M}_{\text{stat}}$), SAC $[256, 256]$ MLP training, and writeup compilation.
- **Bouchet HPC Cluster:** Meta-Llama-3-8B-Instruct inference via `vLLM` with Automatic Prefix Caching (APC) for single-pass logit persona fidelity evaluation and closed-loop training.

