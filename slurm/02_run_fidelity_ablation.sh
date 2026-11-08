#!/bin/bash
# ==============================================================================
# Slurm Job Script: 3-Way Personalization Fidelity Ablation on Yale Bouchet HPC
# Bucket 6 & 7: Persona Alignment Verification (Disentangling Memorization)
# ==============================================================================
#SBATCH --job-name=fidelity_ablation
#SBATCH --account=sds6190
#SBATCH --partition=education_gpu
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-ablation-%j.out
#SBATCH --error=slurm-ablation-%j.err

set -euo pipefail

echo "=========================================================="
echo "Starting Persona Fidelity Ablations on Yale Bouchet Cluster"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Partition:     ${SLURM_JOB_PARTITION:-education_gpu}"
echo "Start Time:    $(date)"
echo "=========================================================="

# 1. Environment & Module Initialization
module purge
module load miniconda || true
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate recsys-rl || conda activate base || echo "Using system Python environment"

# 2. CUDA Diagnostics
echo "--- GPU Hardware Verification ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "GPU check completed"

# 3. Execution of Personalization Ablation Suite across 3 Regimes:
#    (a) Full Persona (\psi_u / psi_u): Empirical history, phi(u), and persona prompt
#    (b) Generic Persona (\psi_{generic} / psi_generic): Neutral instruction without user history
#    (c) Unconditioned (\psi_\emptyset / psi_empty): Bare item metadata without instructions
VLLM_HOST=$(cat artifacts/vllm_host.txt 2>/dev/null || echo "127.0.0.1")
echo "Connecting to vLLM server at host: ${VLLM_HOST}"

echo "--- Running 3-Way Persona Fidelity Evaluation ---"
python scripts/evaluate_fidelity_ablation.py \
    --artifacts-dir artifacts \
    --output-file artifacts/fidelity_ablation_results.json \
    --vllm-url "http://${VLLM_HOST}:8000/v1" \
    --num-users 100 \
    --seed 42

echo "--- Fidelity Ablation Results (artifacts/fidelity_ablation_results.json) ---"
cat artifacts/fidelity_ablation_results.json

echo "=========================================================="
echo "Fidelity Ablation Complete at $(date)"
echo "=========================================================="
