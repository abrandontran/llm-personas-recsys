#!/usr/bin/env python3
"""Script 06: Prepare Bouchet HPC Payload & Pre-Flight Verification.

Bucket 6: Validates that all precomputed artifacts and Slurm scripts are present,
non-zero in size, and correctly formatted for submission to the Yale Bouchet HPC cluster.
Prints explicit cluster payload transfer instructions (rsync / git) and Slurm commands.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("06_prep_bouchet_payload")

# Required artifacts produced by Pre-Bouchet Pipeline (Buckets 1 - 5 & 6)
REQUIRED_ARTIFACTS: List[str] = [
    "data_splits.pkl",
    "item_metadata.json",
    "item_embeddings_d32.npy",
    "user_embeddings_d32.npy",
    "user_archetypes_k32.json",
    "kuairec_matrix.npy",
    "kuairec_embeddings_d32.npy",
    "sac_recsim_baseline.pt",
    "sac_recsim_curve.png",
    "cql_baseline.pt",
    "fidelity_ablation_results.json",
]

# Required Slurm batch scripts
REQUIRED_SLURM_SCRIPTS: List[str] = [
    "01_launch_vllm.sh",
    "02_run_fidelity_ablation.sh",
]

# Essential Slurm directives to check in batch scripts
REQUIRED_SLURM_DIRECTIVES: List[str] = [
    "#SBATCH --partition",
    "#SBATCH --cpus-per-task",
    "#SBATCH --mem",
    "#SBATCH --time",
]


def verify_artifact_payload(
    artifacts_dir: Path,
) -> Tuple[bool, List[Dict[str, str]]]:
    """Verifies that all precomputed pipeline artifacts exist and have non-zero size.

    Args:
        artifacts_dir: Directory containing pipeline artifacts.

    Returns:
        Tuple of (all_passed: bool, details: List[Dict]).
    """
    all_passed = True
    details: List[Dict[str, str]] = []

    for name in REQUIRED_ARTIFACTS:
        file_path = artifacts_dir / name
        if not file_path.exists():
            all_passed = False
            details.append({"name": name, "status": "MISSING", "size": "0 B"})
        elif file_path.stat().st_size == 0:
            all_passed = False
            details.append({"name": name, "status": "EMPTY (0 B)", "size": "0 B"})
        else:
            size_bytes = file_path.stat().st_size
            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.2f} KB"
            else:
                size_str = f"{size_bytes} B"
            details.append({"name": name, "status": "OK", "size": size_str})

    return all_passed, details


def verify_slurm_scripts(
    slurm_dir: Path,
) -> Tuple[bool, List[Dict[str, str]]]:
    """Validates existence, non-zero size, and required headers of Slurm scripts.

    Args:
        slurm_dir: Directory containing Slurm bash scripts.

    Returns:
        Tuple of (all_passed: bool, details: List[Dict]).
    """
    all_passed = True
    details: List[Dict[str, str]] = []

    for script_name in REQUIRED_SLURM_SCRIPTS:
        script_path = slurm_dir / script_name
        if not script_path.exists():
            all_passed = False
            details.append({"script": script_name, "status": "MISSING", "reason": "File does not exist"})
            continue

        if script_path.stat().st_size == 0:
            all_passed = False
            details.append({"script": script_name, "status": "EMPTY", "reason": "0 bytes"})
            continue

        content = script_path.read_text(encoding="utf-8")
        missing_directives = [d for d in REQUIRED_SLURM_DIRECTIVES if d not in content]

        if missing_directives:
            all_passed = False
            details.append({
                "script": script_name,
                "status": "INVALID",
                "reason": f"Missing directives: {', '.join(missing_directives)}",
            })
        else:
            details.append({"script": script_name, "status": "OK", "reason": "All #SBATCH directives valid"})

    return all_passed, details


def print_bouchet_transfer_instructions(
    project_root: Path,
) -> None:
    """Prints clear terminal copy-paste instructions for the Yale Bouchet cluster."""
    print("\n" + "=" * 80)
    print("               YALE BOUCHET HPC CLUSTER TRANSFER & LAUNCH PROTOCOL")
    print("=" * 80)
    print("""
1. Log into Yale Bouchet Cluster via SSH:
   $ ssh <netid>@bouchet.ycrc.yale.edu

2. Transfer Workspace & Artifacts from Local Machine to Bouchet:
   From your local terminal, run rsync to synchronize workspace:
   $ rsync -avz --exclude '.git' --exclude '__pycache__' \\
       ./ <netid>@bouchet.ycrc.yale.edu:~/in-silico-llm-recsys/

3. Configure Environment on Bouchet:
   $ cd ~/in-silico-llm-recsys
   $ module load miniconda
   $ conda create -n recsys-rl python=3.10 -y && conda activate recsys-rl
   $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   $ pip install vllm gymnasium scipy scikit-learn matplotlib

4. Submit Job 1: High-Throughput vLLM Server (Single GPU, APC Prefix Caching):
   $ sbatch slurm/01_launch_vllm.sh
   # Monitor server readiness:
   $ squeue -u $USER
   $ tail -f slurm-vllm-*.out

5. Submit Job 2: 3-Way Personalization Fidelity Ablation Suite:
   $ sbatch slurm/02_run_fidelity_ablation.sh
   # View completed ablation metrics:
   $ cat artifacts/fidelity_ablation_results.json

6. Optional: Set Up Local SSH Tunnel to Bouchet vLLM Server:
   From local machine:
   $ ssh -N -L 8000:localhost:8000 <netid>@bouchet.ycrc.yale.edu
   # Now local MockLLMEnv or client can query Bouchet vLLM at http://localhost:8000/v1
""")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Pre-Bouchet artifacts and Slurm scripts for HPC deployment."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Path to artifacts directory (default: artifacts)",
    )
    parser.add_argument(
        "--slurm-dir",
        type=str,
        default="slurm",
        help="Path to slurm directory (default: slurm)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any verification check fails.",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    slurm_dir = Path(args.slurm_dir).resolve()
    project_root = artifacts_dir.parent

    logger.info("=== Pre-Bouchet HPC Payload Verification (Bucket 6) ===")

    # 1. Verify Pipeline Artifacts
    artifacts_ok, artifact_details = verify_artifact_payload(artifacts_dir)
    print("\n--- Artifact Verification ---")
    print(f"{'Artifact File':<36} | {'Status':<12} | {'Size':<12}")
    print("-" * 65)
    for item in artifact_details:
        print(f"{item['name']:<36} | {item['status']:<12} | {item['size']:<12}")

    # 2. Verify Slurm Batch Scripts
    slurm_ok, slurm_details = verify_slurm_scripts(slurm_dir)
    print("\n--- Slurm Scripts Verification ---")
    print(f"{'Script Name':<32} | {'Status':<8} | {'Details'}")
    print("-" * 75)
    for item in slurm_details:
        print(f"{item['script']:<32} | {item['status']:<8} | {item['reason']}")

    all_verified = artifacts_ok and slurm_ok

    if all_verified:
        logger.info("[PASSED] All artifacts and Slurm scripts verified successfully!")
        print_bouchet_transfer_instructions(project_root)
        sys.exit(0)
    else:
        logger.error("[FAILED] Verification failed! One or more required files are missing or invalid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
