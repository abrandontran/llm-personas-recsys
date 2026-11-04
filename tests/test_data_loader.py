import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.data.loader import (
    generate_enriched_synopsis,
    split_user_interactions,
    download_and_load_movielens,
    download_and_load_kuairec,
)


class TestDataLoader(unittest.TestCase):
    def test_synopsis_generator(self):
        # Diverse genres and decades
        synopsis_toy = generate_enriched_synopsis(
            "Toy Story", 1995, ["Animation", "Children's", "Comedy"]
        )
        self.assertIsInstance(synopsis_toy, str)
        self.assertIn("1990s", synopsis_toy)
        self.assertIn("animation", synopsis_toy.lower())
        self.assertTrue(synopsis_toy.endswith("."))

        synopsis_godfather = generate_enriched_synopsis(
            "The Godfather", 1972, ["Crime", "Drama"]
        )
        self.assertIsInstance(synopsis_godfather, str)
        self.assertIn("1970s", synopsis_godfather)
        self.assertIn("crime", synopsis_godfather.lower())
        self.assertTrue(synopsis_godfather.endswith("."))

        synopsis_scifi = generate_enriched_synopsis(
            "Minority Report", 2002, ["Sci-Fi", "Action", "Thriller"]
        )
        self.assertIsInstance(synopsis_scifi, str)
        self.assertIn("2000s", synopsis_scifi)
        self.assertIn("sci-fi", synopsis_scifi.lower())

        # Determinism check
        synopsis_repeat = generate_enriched_synopsis(
            "Toy Story", 1995, ["Animation", "Children's", "Comedy"]
        )
        self.assertEqual(synopsis_toy, synopsis_repeat)

        # Fallback for empty/uncommon genres
        synopsis_empty = generate_enriched_synopsis("Mystery Film", 1985, [])
        self.assertIsInstance(synopsis_empty, str)
        self.assertIn("1980s", synopsis_empty)

    def test_chronological_split(self):
        # 10 interactions with unordered timestamps
        items = list(range(10))
        ratings = [4.0] * 10
        timestamps = [105, 101, 109, 100, 108, 102, 107, 103, 106, 104]

        train, eval_set = split_user_interactions(items, ratings, timestamps, train_ratio=0.6)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(eval_set), 4)

        # Ensure train timestamps are sorted
        train_ts = [t[2] for t in train]
        eval_ts = [t[2] for t in eval_set]
        self.assertEqual(train_ts, sorted(train_ts))
        self.assertEqual(eval_ts, sorted(eval_ts))
        # Ensure chronological partition boundary: max(train) <= min(eval)
        self.assertLessEqual(max(train_ts), min(eval_ts))

        # Check interaction tuple format (item_id, rating, timestamp)
        for interaction in train + eval_set:
            self.assertEqual(len(interaction), 3)

    def test_synthetic_movielens_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "data"
            artifacts_dir = tmp_path / "artifacts"
            ml_dir = data_dir / "ml-1m"
            ml_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Create synthetic movies.dat
            movies_dat = ml_dir / "movies.dat"
            with open(movies_dat, "w", encoding="latin-1") as f:
                f.write("1::Toy Story (1995)::Animation|Children's|Comedy\n")
                f.write("2::Jumanji (1995)::Adventure|Children's|Fantasy\n")
                f.write("3::Heat (1995)::Action|Crime|Thriller\n")

            # Create synthetic ratings.dat
            # User 1 has 55 ratings (>= 50 min_ratings threshold)
            # User 2 has 10 ratings (< 50 min_ratings threshold)
            ratings_dat = ml_dir / "ratings.dat"
            with open(ratings_dat, "w", encoding="utf-8") as f:
                for i in range(55):
                    movie_id = (i % 3) + 1
                    f.write(f"1::{movie_id}::4.0::{1000 + i}\n")
                for i in range(10):
                    movie_id = (i % 3) + 1
                    f.write(f"2::{movie_id}::3.0::{2000 + i}\n")

            splits = download_and_load_movielens(
                data_dir=data_dir, artifacts_dir=artifacts_dir, min_ratings=50
            )

            # Verify filtering: User 1 is present, User 2 is filtered out
            self.assertIn("train_interactions", splits)
            self.assertIn("eval_interactions", splits)
            self.assertIn(1, splits["user_ids"])
            self.assertNotIn(2, splits["user_ids"])

            # Verify chronological split on User 1: 55 * 0.6 = 33 train, 22 eval
            self.assertEqual(len(splits["train_interactions"][1]), 33)
            self.assertEqual(len(splits["eval_interactions"][1]), 22)

            # Verify persisted data_splits.pkl
            pkl_path = artifacts_dir / "data_splits.pkl"
            self.assertTrue(pkl_path.exists())
            with open(pkl_path, "rb") as f:
                saved_splits = pickle.load(f)
            self.assertIn(1, saved_splits["train_interactions"])

            # Verify persisted item_metadata.json
            meta_path = artifacts_dir / "item_metadata.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path, "r", encoding="utf-8") as f:
                item_metadata = json.load(f)

            self.assertIn("1", item_metadata)
            meta_1 = item_metadata["1"]
            self.assertEqual(meta_1["title"], "Toy Story")
            self.assertEqual(meta_1["year"], 1995)
            self.assertEqual(meta_1["genres"], ["Animation", "Children's", "Comedy"])
            self.assertIn("1990s", meta_1["synopsis"])

    def test_kuairec_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "data"
            artifacts_dir = tmp_path / "artifacts"

            matrix = download_and_load_kuairec(data_dir=data_dir, artifacts_dir=artifacts_dir)
            self.assertIsInstance(matrix, np.ndarray)
            self.assertEqual(matrix.shape, (1411, 3327))

            npy_path = artifacts_dir / "kuairec_matrix.npy"
            self.assertTrue(npy_path.exists())
            saved_matrix = np.load(npy_path)
            self.assertEqual(saved_matrix.shape, (1411, 3327))


if __name__ == "__main__":
    unittest.main()
