"""Unit tests for SVD latent factorization, RMSE validation, and KuaiRec embeddings."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.data.embeddings import (
    build_rating_matrix,
    fit_kuairec_svd,
    fit_svd_embeddings,
    run_embeddings_pipeline,
    validate_svd_rmse,
)


class TestEmbeddings(unittest.TestCase):
    """Test suite for SVD factorization and embedding pipelines."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_build_rating_matrix(self):
        """Test sparse CSR matrix construction from user interactions."""
        user_ids = [10, 20, 30]
        item_ids = [100, 200, 300, 400]
        train_interactions = {
            10: [(100, 4.0, 1000), (300, 5.0, 1001)],
            20: [(200, 3.5, 1002)],
            30: [(400, 2.0, 1003), (100, 1.0, 1004)],
        }

        R = build_rating_matrix(train_interactions, user_ids, item_ids)

        self.assertTrue(sp.isspmatrix_csr(R))
        self.assertEqual(R.shape, (3, 4))
        self.assertEqual(R.dtype, np.float32)

        # Verify entry coordinates and values
        self.assertEqual(R[0, 0], 4.0)
        self.assertEqual(R[0, 2], 5.0)
        self.assertEqual(R[1, 1], 3.5)
        self.assertEqual(R[2, 3], 2.0)
        self.assertEqual(R[2, 0], 1.0)
        self.assertEqual(R[0, 1], 0.0)

    def test_svd_dimensions_and_norm(self):
        """Test SVD factorization output dimensions, centering, and unit normalization."""
        # Synthetic user-item matrix: 100 users, 50 items
        num_users, num_items, d = 100, 50, 32
        R_dense = self.rng.choice(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            size=(num_users, num_items),
            p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.1],
        ).astype(np.float32)
        R = sp.csr_matrix(R_dense)

        u_emb, e_norm, e_raw, user_means = fit_svd_embeddings(R, d=d)

        # Output shapes
        self.assertEqual(u_emb.shape, (num_users, d))
        self.assertEqual(e_norm.shape, (num_items, d))
        self.assertEqual(e_raw.shape, (num_items, d))
        self.assertEqual(user_means.shape, (num_users,))

        # Data types
        self.assertEqual(u_emb.dtype, np.float32)
        self.assertEqual(e_norm.dtype, np.float32)
        self.assertEqual(e_raw.dtype, np.float32)
        self.assertEqual(user_means.dtype, np.float32)

        # Unit norm check on items with non-zero vectors
        norms = np.linalg.norm(e_norm, axis=1)
        active_items = np.linalg.norm(e_raw, axis=1) > 1e-6
        np.testing.assert_allclose(norms[active_items], 1.0, rtol=1e-5)

    def test_validate_svd_rmse(self):
        """Test RMSE validation calculation on synthetic train/eval splits."""
        num_users, num_items, d = 120, 60, 16
        user_ids = list(range(1, num_users + 1))
        item_ids = list(range(101, 101 + num_items))

        # Synthetic latent factors
        u_latent = self.rng.normal(0, 0.3, size=(num_users, d)).astype(np.float32)
        v_latent = self.rng.normal(0, 0.3, size=(num_items, d)).astype(np.float32)
        base_means = self.rng.uniform(3.0, 3.8, size=(num_users,)).astype(np.float32)

        train_inters = {}
        eval_inters = {}

        for u_idx, uid in enumerate(user_ids):
            # Each user rates 40 items total (24 train, 16 eval)
            chosen_items = self.rng.choice(item_ids, size=40, replace=False)
            train_items = chosen_items[:24]
            eval_items = chosen_items[24:]

            train_list = []
            for iid in train_items:
                i_idx = iid - 101
                true_r = base_means[u_idx] + np.dot(u_latent[u_idx], v_latent[i_idx])
                noisy_r = float(np.clip(true_r + self.rng.normal(0, 0.8), 1.0, 5.0))
                train_list.append((iid, noisy_r, 100))
            train_inters[uid] = train_list

            eval_list = []
            for iid in eval_items:
                i_idx = iid - 101
                true_r = base_means[u_idx] + np.dot(u_latent[u_idx], v_latent[i_idx])
                noisy_r = float(np.clip(true_r + self.rng.normal(0, 0.85), 1.0, 5.0))
                eval_list.append((iid, noisy_r, 200))
            eval_inters[uid] = eval_list

        R = build_rating_matrix(train_inters, user_ids, item_ids)
        u_emb, e_norm, e_raw, user_means = fit_svd_embeddings(R, d=d)

        rmse = validate_svd_rmse(
            train_inters,
            eval_inters,
            u_emb,
            e_raw,
            user_means,
            user_ids,
            item_ids,
        )

        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0.75)
        self.assertLessEqual(rmse, 1.05)

    def test_validate_svd_rmse_sparse_default_means(self):
        """Test validate_svd_rmse with scipy.sparse matrices and default user_means=None."""
        num_users, num_items, d = 100, 50, 16
        u_latent = self.rng.normal(0, 0.3, size=(num_users, d)).astype(np.float32)
        v_latent = self.rng.normal(0, 0.3, size=(num_items, d)).astype(np.float32)
        base_means = self.rng.uniform(3.0, 3.8, size=(num_users,)).astype(np.float32)

        R_dense = np.zeros((num_users, num_items), dtype=np.float32)
        for u in range(num_users):
            for i in range(num_items):
                if self.rng.random() > 0.6:
                    r = base_means[u] + np.dot(u_latent[u], v_latent[i]) + self.rng.normal(0, 0.8)
                    R_dense[u, i] = np.clip(r, 1.0, 5.0)

        mask_train = (R_dense > 0) & (self.rng.random(size=(num_users, num_items)) > 0.4)
        mask_eval = (R_dense > 0) & (~mask_train)

        R_train = sp.csr_matrix(R_dense * mask_train)
        R_eval = sp.csr_matrix(R_dense * mask_eval)

        u_emb, e_norm, e_raw, user_means = fit_svd_embeddings(R_train, d=d)

        # Call with default user_means=None and sparse matrices
        rmse_default = validate_svd_rmse(R_train, R_eval, u_emb, e_raw)

        self.assertIsInstance(rmse_default, float)
        self.assertGreaterEqual(rmse_default, 0.75)
        self.assertLessEqual(rmse_default, 1.05)

        # Compare against explicit user_means call
        rmse_explicit = validate_svd_rmse(
            R_train, R_eval, u_emb, e_raw, user_means=user_means
        )
        self.assertAlmostEqual(rmse_default, rmse_explicit, places=5)

    def test_kuairec_svd(self):
        """Test KuaiRec SVD on synthetic (100, 50) dense matrix."""
        num_users, num_items, d = 100, 50, 32
        dense_matrix = self.rng.uniform(0.0, 1.0, size=(num_users, num_items)).astype(np.float32)

        kuairec_embs = fit_kuairec_svd(dense_matrix, d=d)

        self.assertEqual(kuairec_embs.shape, (num_items, d))
        self.assertEqual(kuairec_embs.dtype, np.float32)

        # Check unit normalization
        norms = np.linalg.norm(kuairec_embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_run_embeddings_pipeline_synthetic(self):
        """Test end-to-end embeddings pipeline on synthetic artifacts directory."""
        num_users, num_items, d = 80, 40, 16
        user_ids = list(range(1, num_users + 1))
        item_ids = list(range(1001, 1001 + num_items))

        train_inters = {
            uid: [(item_ids[i % num_items], float((i % 5) + 1), 100) for i in range(25)]
            for uid in user_ids
        }
        eval_inters = {
            uid: [(item_ids[(i + 5) % num_items], float((i % 5) + 1), 200) for i in range(15)]
            for uid in user_ids
        }

        splits_payload = {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "train_interactions": train_inters,
            "eval_interactions": eval_inters,
        }

        kuairec_dense = self.rng.uniform(0.0, 1.0, size=(100, 50)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            with open(artifacts_dir / "data_splits.pkl", "wb") as f:
                pickle.dump(splits_payload, f)
            np.save(artifacts_dir / "kuairec_matrix.npy", kuairec_dense)

            res = run_embeddings_pipeline(
                artifacts_dir=artifacts_dir,
                d=d,
                min_rmse=None,
                max_rmse=None,
            )

            self.assertIn("validation_rmse", res)
            self.assertEqual(res["user_embeddings_shape"], (num_users, d))
            self.assertEqual(res["item_embeddings_shape"], (num_items, d))
            self.assertEqual(res["kuairec_embeddings_shape"], (50, d))

            # Verify saved files
            item_path = artifacts_dir / f"item_embeddings_d{d}.npy"
            user_path = artifacts_dir / f"user_embeddings_d{d}.npy"
            kuai_path = artifacts_dir / f"kuairec_embeddings_d{d}.npy"

            self.assertTrue(item_path.exists())
            self.assertTrue(user_path.exists())
            self.assertTrue(kuai_path.exists())

            saved_item = np.load(item_path)
            saved_user = np.load(user_path)
            saved_kuai = np.load(kuai_path)

            self.assertEqual(saved_item.shape, (num_items, d))
            self.assertEqual(saved_user.shape, (num_users, d))
            self.assertEqual(saved_kuai.shape, (50, d))


if __name__ == "__main__":
    unittest.main()
