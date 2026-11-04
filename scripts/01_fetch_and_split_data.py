#!/usr/bin/env python3
"""Script 01: Fetch datasets, compute chronological splits, generate item metadata and KuaiRec matrix.

Bucket 1 of the Pre-Bouchet Pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import download_and_load_kuairec, download_and_load_movielens

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("01_fetch_and_split_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MovieLens-1M and KuaiRec data, generate chronological splits and rich metadata."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store downloaded raw datasets (default: data)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to store processed artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=50,
        help="Minimum number of user ratings for filtering (default: 50)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)

    logger.info("=== Starting Bucket 1: Data Ingestion & Chronological Splitting ===")
    t_start = time.time()

    # Step 1: MovieLens-1M
    logger.info(f"Loading MovieLens-1M (min_ratings={args.min_ratings})...")
    t0 = time.time()
    splits = download_and_load_movielens(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        min_ratings=args.min_ratings,
    )
    t_ml = time.time() - t0
    num_users = len(splits["user_ids"])
    num_items = len(splits["item_ids"])
    num_train = sum(len(v) for v in splits["train_interactions"].values())
    num_eval = sum(len(v) for v in splits["eval_interactions"].values())
    logger.info(
        f"MovieLens-1M loaded in {t_ml:.2f}s: {num_users} users, {num_items} items, "
        f"{num_train} train interactions, {num_eval} eval interactions."
    )

    # Step 2: KuaiRec
    logger.info("Loading KuaiRec small matrix benchmark...")
    t1 = time.time()
    kuairec_matrix = download_and_load_kuairec(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    t_kuai = time.time() - t1
    density = (kuairec_matrix != 0).mean() * 100.0
    logger.info(
        f"KuaiRec loaded in {t_kuai:.2f}s: shape {kuairec_matrix.shape}, "
        f"observed density {density:.2f}%."
    )

    total_elapsed = time.time() - t_start
    logger.info(
        f"=== Bucket 1 Completed Successfully in {total_elapsed:.2f}s (Budget: <30s) ==="
    )
    logger.info(f"Artifacts generated:")
    logger.info(f"  - {artifacts_dir / 'data_splits.pkl'}")
    logger.info(f"  - {artifacts_dir / 'item_metadata.json'}")
    logger.info(f"  - {artifacts_dir / 'kuairec_matrix.npy'}")


if __name__ == "__main__":
    main()
