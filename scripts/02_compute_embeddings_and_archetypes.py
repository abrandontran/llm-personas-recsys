#!/usr/bin/env python3
"""Script 02: Compute SVD latent embeddings, validate RMSE, compute behavioral signatures phi(u),
and cluster k=32 user archetypes.

Chains Bucket 2a (Embeddings) and Bucket 2b (Archetypes) of the Pre-Bouchet Pipeline.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.archetypes import run_archetypes_pipeline
from src.data.embeddings import run_embeddings_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("02_compute_embeddings_and_archetypes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SVD latent embeddings and cluster k=32 user archetypes."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory storing artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=32,
        help="Latent factor embedding dimension (default: 32)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of user archetypes to cluster (default: 32)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for k-means clustering (default: 42)",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Force recomputation of SVD embeddings even if existing files are present.",
    )
    parser.add_argument(
        "--min-rmse",
        type=float,
        default=0.75,
        help="Minimum acceptable SVD validation RMSE bound (default: 0.75)",
    )
    parser.add_argument(
        "--no-normalize-blocks",
        action="store_true",
        help="Disable block variance balancing prior to k-means clustering.",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=1.05,
        help="Maximum acceptable SVD validation RMSE bound (default: 1.05)",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    logger.info("=== Starting Bucket 2: Embeddings & Archetypes Pipeline ===")
    t_start = time.time()

    # Step 1: Check or compute SVD latent embeddings (Bucket 2a)
    user_emb_path = artifacts_dir / f"user_embeddings_d{args.d}.npy"
    item_emb_path = artifacts_dir / f"item_embeddings_d{args.d}.npy"
    kuai_emb_path = artifacts_dir / f"kuairec_embeddings_d{args.d}.npy"

    if (
        args.recompute_embeddings
        or not user_emb_path.exists()
        or not item_emb_path.exists()
        or not kuai_emb_path.exists()
    ):
        logger.info(f"Running SVD matrix factorization pipeline (d={args.d})...")
        t_emb_start = time.time()
        emb_results = run_embeddings_pipeline(
            artifacts_dir=artifacts_dir,
            d=args.d,
            min_rmse=args.min_rmse,
            max_rmse=args.max_rmse,
        )
        t_emb = time.time() - t_emb_start
        logger.info(
            f"Embeddings pipeline completed in {t_emb:.2f}s: "
            f"Validation RMSE = {emb_results['validation_rmse']:.4f}."
        )
    else:
        logger.info(f"Found existing embeddings at {artifacts_dir}. Skipping factorization.")

    # Step 2: Behavioral Signatures & Archetype Extraction (Bucket 2b)
    logger.info(
        f"Extracting k={args.k} archetypes from behavioral signatures phi(u) in R^21..."
    )
    t_arc_start = time.time()
    archetypes = run_archetypes_pipeline(
        artifacts_dir=artifacts_dir,
        d=args.d,
        k=args.k,
        random_state=args.random_state,
        normalize_blocks=not args.no_normalize_blocks,
    )
    t_arc = time.time() - t_arc_start
    logger.info(f"Archetype extraction completed in {t_arc:.2f}s.")

    # Step 3: Print summary statistics
    tau_quits = [arc["tau_quit"] for arc in archetypes]
    mean_tau = sum(tau_quits) / len(tau_quits)
    min_tau = min(tau_quits)
    max_tau = max(tau_quits)

    total_elapsed = time.time() - t_start
    logger.info("=== Bucket 2 Summary Statistics ===")
    logger.info(f"Archetypes generated: {len(archetypes)}")
    logger.info(f"Quit threshold tau_quit: mean={mean_tau:.2f}, min={min_tau:.2f}, max={max_tau:.2f}")

    logger.info("Sample Archetype 0:")
    sample = archetypes[0]
    logger.info(f"  Archetype ID: {sample['archetype_id']}")
    logger.info(f"  Exemplar User ID: {sample['user_id']}")
    logger.info(f"  Top Genres: {sample['top_genres']}")
    logger.info(f"  Favorite Movies: {sample['favorite_movies']}")
    logger.info(f"  tau_quit: {sample['tau_quit']}")
    logger.info(f"  System Prompt Preview:\n    " + sample["system_prompt"].replace("\n", "\n    "))

    logger.info(
        f"=== Bucket 2 Completed Successfully in {total_elapsed:.2f}s (Budget: <30s) ==="
    )
    logger.info(f"Artifacts verified:")
    logger.info(f"  - {user_emb_path}")
    logger.info(f"  - {item_emb_path}")
    logger.info(f"  - {kuai_emb_path}")
    logger.info(f"  - {artifacts_dir / f'user_archetypes_k{args.k}.json'}")


if __name__ == "__main__":
    main()
