"""SVD latent matrix factorization, RMSE validation check, and KuaiRec item embeddings.

Implements Bucket 2a of the Pre-Bouchet Pipeline.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

logger = logging.getLogger(__name__)


def build_rating_matrix(
    train_interactions: Dict[Any, List[Any]],
    user_ids: List[Any],
    item_ids: List[Any],
) -> sp.csr_matrix:
    """Builds a sparse CSR rating matrix R of shape (|U|, |I|) containing user ratings.

    Args:
        train_interactions: Mapping from user_id to list of interaction tuples.
            Tuples can be (item_id, rating) or (item_id, rating, timestamp, ...).
        user_ids: List of canonical user identifiers defining row order.
        item_ids: List of canonical item identifiers defining column order.

    Returns:
        sp.csr_matrix of shape (|U|, |I|) with float32 rating entries.
    """
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    rows: List[int] = []
    cols: List[int] = []
    ratings: List[float] = []

    for uid, inters in train_interactions.items():
        if uid not in user_to_idx:
            continue
        u_idx = user_to_idx[uid]
        for inter in inters:
            iid = inter[0]
            rating = float(inter[1])
            if iid in item_to_idx:
                rows.append(u_idx)
                cols.append(item_to_idx[iid])
                ratings.append(rating)

    R = sp.csr_matrix(
        (ratings, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    return R


def fit_svd_embeddings(
    R: Union[sp.spmatrix, np.ndarray],
    d: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits Truncated SVD on user-mean-centered observed ratings.

    Subtracts user mean ratings on non-zero entries (centering) to avoid bias
    toward user baseline rating volume.

    Args:
        R: Rating matrix, either sparse matrix or 2D dense numpy array (|U|, |I|).
        d: Latent dimensionality (default: 32).

    Returns:
        tuple containing:
            - u_emb: User latent factors (|U|, d), U * sqrt(Sigma)
            - e_norm: Unit-normalized item embeddings (|I|, d)
            - e_raw: Unnormalized item latent factors (|I|, d), V * sqrt(Sigma)
            - user_means: 1D array of per-user observed rating means (|U|,)
    """
    if not sp.isspmatrix_csr(R):
        if sp.issparse(R):
            R_sparse = R.tocsr().astype(np.float32, copy=True)
        else:
            R_sparse = sp.csr_matrix(R, dtype=np.float32)
    else:
        R_sparse = R.copy().astype(np.float32)

    num_users, num_items = R_sparse.shape

    # Compute per-user means on non-zero observed entries
    user_means = np.zeros(num_users, dtype=np.float32)
    global_mean = float(np.mean(R_sparse.data)) if len(R_sparse.data) > 0 else 0.0

    R_centered = R_sparse.copy()
    for u in range(num_users):
        start = R_centered.indptr[u]
        end = R_centered.indptr[u + 1]
        if end > start:
            m = float(np.mean(R_centered.data[start:end]))
            user_means[u] = m
            R_centered.data[start:end] -= m
        else:
            user_means[u] = global_mean

    # Fit Truncated SVD
    min_dim = min(num_users, num_items)
    if d < min_dim:
        U, s, Vt = sla.svds(R_centered, k=d, which="LM")
        # svds returns singular values in ascending order; sort descending
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
    else:
        # Fallback for small synthetic dimensions
        dense_centered = R_centered.toarray()
        U_full, s_full, Vt_full = np.linalg.svd(dense_centered, full_matrices=False)
        k = min(d, len(s_full))
        U = np.zeros((num_users, d), dtype=np.float32)
        Vt = np.zeros((d, num_items), dtype=np.float32)
        s = np.zeros(d, dtype=np.float32)

        U[:, :k] = U_full[:, :k]
        s[:k] = s_full[:k]
        Vt[:k, :] = Vt_full[:k, :]

    sqrt_s = np.sqrt(s).astype(np.float32)
    u_emb = (U * sqrt_s).astype(np.float32)
    e_raw = (Vt.T * sqrt_s).astype(np.float32)

    # Unit-normalize item embeddings
    norms = np.linalg.norm(e_raw, axis=1, keepdims=True) + 1e-12
    e_norm = (e_raw / norms).astype(np.float32)

    return u_emb, e_norm, e_raw, user_means


def compute_user_means(
    train_interactions: Any,
    num_users: int,
    user_ids: Optional[List[Any]] = None,
) -> np.ndarray:
    """Computes mean observed rating per user from training interactions.

    Args:
        train_interactions: Sparse matrix, dense array, or dict of user interactions.
        num_users: Total number of users.
        user_ids: Optional list of user identifiers if train_interactions is a dict.

    Returns:
        np.ndarray of shape (num_users,) with per-user mean ratings (float32).
    """
    user_means = np.zeros(num_users, dtype=np.float32)
    has_ratings = np.zeros(num_users, dtype=bool)
    all_ratings: List[float] = []

    if isinstance(train_interactions, dict):
        if user_ids is not None:
            u_map = {uid: idx for idx, uid in enumerate(user_ids)}
        else:
            u_map = {uid: uid for uid in train_interactions.keys()}

        for uid, inters in train_interactions.items():
            if uid not in u_map:
                continue
            u_idx = u_map[uid]
            if 0 <= u_idx < num_users and len(inters) > 0:
                ratings = [float(x[1]) for x in inters]
                user_means[u_idx] = float(np.mean(ratings))
                has_ratings[u_idx] = True
                all_ratings.extend(ratings)
        global_mean = float(np.mean(all_ratings)) if all_ratings else 0.0
    else:
        if not sp.isspmatrix_csr(train_interactions):
            if sp.issparse(train_interactions):
                csr = train_interactions.tocsr()
            else:
                csr = sp.csr_matrix(train_interactions, dtype=np.float32)
        else:
            csr = train_interactions

        global_mean = float(np.mean(csr.data)) if len(csr.data) > 0 else 0.0
        for u in range(num_users):
            start = csr.indptr[u]
            end = csr.indptr[u + 1]
            if end > start:
                user_means[u] = float(np.mean(csr.data[start:end]))
                has_ratings[u] = True

    user_means[~has_ratings] = global_mean
    return user_means


def validate_svd_rmse(
    train_interactions: Any,
    eval_interactions: Any,
    u_emb: np.ndarray,
    e_raw: np.ndarray,
    user_means: Optional[np.ndarray] = None,
    user_ids: Optional[List[Any]] = None,
    item_ids: Optional[List[Any]] = None,
    min_rmse: Optional[float] = 0.75,
    max_rmse: Optional[float] = 1.05,
) -> float:
    """Predicts ratings on observed evaluation pairs and validates reconstruction RMSE.

    Rating prediction formula:
        r_hat_{u, i} = r_bar_u + u_u.T @ e_i (clamped to [1.0, 5.0])
    RMSE:
        sqrt(1/N * sum((r_{u, i} - r_hat_{u, i})^2))

    Asserts that RMSE lies within [min_rmse, max_rmse] (target benchmark ~ 0.874).

    Args:
        train_interactions: Interaction dict or sparse matrix for train set.
        eval_interactions: Interaction dict or sparse matrix for eval set.
        u_emb: User embeddings (|U|, d).
        e_raw: Unnormalized item embeddings (|I|, d).
        user_means: Per-user mean ratings. If None, derived from train_interactions.
        user_ids: List of canonical user IDs.
        item_ids: List of canonical item IDs.
        min_rmse: Minimum expected RMSE assertion bound (default: 0.75).
        max_rmse: Maximum expected RMSE assertion bound (default: 1.05).

    Returns:
        float validation RMSE.
    """
    if isinstance(eval_interactions, dict):
        if user_ids is None or item_ids is None:
            raise ValueError("user_ids and item_ids are required when eval_interactions is a dict.")
        u_map = {uid: idx for idx, uid in enumerate(user_ids)}
        i_map = {iid: idx for idx, iid in enumerate(item_ids)}

        u_indices: List[int] = []
        i_indices: List[int] = []
        actual_ratings: List[float] = []

        for uid, inters in eval_interactions.items():
            if uid not in u_map:
                continue
            u_idx = u_map[uid]
            for inter in inters:
                iid = inter[0]
                rating = float(inter[1])
                if iid in i_map:
                    u_indices.append(u_idx)
                    i_indices.append(i_map[iid])
                    actual_ratings.append(rating)

        u_arr = np.array(u_indices, dtype=np.int32)
        i_arr = np.array(i_indices, dtype=np.int32)
        r_arr = np.array(actual_ratings, dtype=np.float32)
    elif sp.issparse(eval_interactions):
        coo = eval_interactions.tocoo()
        u_arr = coo.row.astype(np.int32)
        i_arr = coo.col.astype(np.int32)
        r_arr = coo.data.astype(np.float32)
    elif isinstance(eval_interactions, np.ndarray):
        rows, cols = np.nonzero(eval_interactions)
        u_arr = rows.astype(np.int32)
        i_arr = cols.astype(np.int32)
        r_arr = eval_interactions[rows, cols].astype(np.float32)
    else:
        raise TypeError(f"Unsupported eval_interactions type: {type(eval_interactions)}")

    if len(r_arr) == 0:
        raise ValueError("No observed evaluation pairs found for RMSE validation.")

    if user_means is None:
        user_means = compute_user_means(
            train_interactions=train_interactions,
            num_users=u_emb.shape[0],
            user_ids=user_ids,
        )

    u_vecs = u_emb[u_arr]
    e_vecs = e_raw[i_arr]
    u_m = user_means[u_arr]

    dot_products = np.sum(u_vecs * e_vecs, axis=1)
    preds = u_m + dot_products
    preds = np.clip(preds, 1.0, 5.0)

    sq_errors = (r_arr - preds) ** 2
    rmse = float(np.sqrt(np.mean(sq_errors)))

    logger.info(
        f"SVD RMSE validation on {len(r_arr)} evaluation interactions: {rmse:.4f}"
    )

    if min_rmse is not None and max_rmse is not None:
        assert min_rmse <= rmse <= max_rmse, (
            f"Validation RMSE {rmse:.4f} is outside expected range [{min_rmse}, {max_rmse}] "
            f"(target benchmark in proposal is 0.874)."
        )

    return rmse


def fit_kuairec_svd(kuairec_dense_matrix: np.ndarray, d: int = 32) -> np.ndarray:
    """Performs Truncated SVD on KuaiRec dense matrix and extracts unit-normalized item embeddings.

    Args:
        kuairec_dense_matrix: Dense numpy array of shape (|U_kuai|, |I_kuai|), e.g. (1411, 3327).
        d: Latent embedding dimension (default: 32).

    Returns:
        np.ndarray of shape (|I_kuai|, d) with float32 unit-normalized item embeddings.
    """
    matrix = np.asarray(kuairec_dense_matrix, dtype=np.float32)
    num_users, num_items = matrix.shape

    min_dim = min(num_users, num_items)
    if d < min_dim:
        U, s, Vt = sla.svds(matrix, k=d, which="LM")
        idx = np.argsort(s)[::-1]
        s = s[idx]
        Vt = Vt[idx, :]
    else:
        U_full, s_full, Vt_full = np.linalg.svd(matrix, full_matrices=False)
        k = min(d, len(s_full))
        Vt = np.zeros((d, num_items), dtype=np.float32)
        s = np.zeros(d, dtype=np.float32)
        s[:k] = s_full[:k]
        Vt[:k, :] = Vt_full[:k, :]

    sqrt_s = np.sqrt(s).astype(np.float32)
    e_raw = (Vt.T * sqrt_s).astype(np.float32)

    norms = np.linalg.norm(e_raw, axis=1, keepdims=True) + 1e-12
    kuairec_embs = (e_raw / norms).astype(np.float32)

    return kuairec_embs


def run_embeddings_pipeline(
    artifacts_dir: Union[str, Path] = Path("artifacts"),
    d: int = 32,
    min_rmse: Optional[float] = 0.75,
    max_rmse: Optional[float] = 1.05,
) -> Dict[str, Any]:
    """Runs the complete embeddings pipeline: MovieLens SVD, RMSE validation, and KuaiRec SVD.

    Produces and persists:
        - `artifacts_dir / f"item_embeddings_d{d}.npy"` (|I| x d float32 unit-normalized)
        - `artifacts_dir / f"user_embeddings_d{d}.npy"` (|U| x d float32)
        - `artifacts_dir / f"kuairec_embeddings_d{d}.npy"` (3327 x d float32 unit-normalized)

    Args:
        artifacts_dir: Directory containing input artifacts and destination for embeddings.
        d: Latent embedding dimension (default: 32).
        min_rmse: Minimum expected RMSE assertion bound (default: 0.75).
        max_rmse: Maximum expected RMSE assertion bound (default: 1.05).

    Returns:
        dict with output shapes, validation RMSE, and statistics.
    """
    artifacts_dir = Path(artifacts_dir)
    data_splits_path = artifacts_dir / "data_splits.pkl"
    kuairec_matrix_path = artifacts_dir / "kuairec_matrix.npy"

    if not data_splits_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {data_splits_path}")
    if not kuairec_matrix_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {kuairec_matrix_path}")

    logger.info(f"Loading data splits from {data_splits_path}...")
    with open(data_splits_path, "rb") as f:
        splits = pickle.load(f)

    user_ids = splits["user_ids"]
    item_ids = splits["item_ids"]
    train_interactions = splits["train_interactions"]
    eval_interactions = splits["eval_interactions"]

    logger.info(
        f"Building rating matrix for {len(user_ids)} users and {len(item_ids)} items..."
    )
    R = build_rating_matrix(train_interactions, user_ids, item_ids)

    logger.info(f"Fitting SVD factorization (d={d}) on MovieLens-1M...")
    u_emb, e_norm, e_raw, user_means = fit_svd_embeddings(R, d=d)

    logger.info("Validating reconstruction RMSE on chronological eval interactions...")
    rmse = validate_svd_rmse(
        train_interactions=train_interactions,
        eval_interactions=eval_interactions,
        u_emb=u_emb,
        e_raw=e_raw,
        user_means=user_means,
        user_ids=user_ids,
        item_ids=item_ids,
        min_rmse=min_rmse,
        max_rmse=max_rmse,
    )

    logger.info(f"Loading KuaiRec matrix from {kuairec_matrix_path}...")
    kuairec_matrix = np.load(kuairec_matrix_path)

    logger.info(f"Fitting SVD on KuaiRec matrix (shape {kuairec_matrix.shape}, d={d})...")
    kuairec_embs = fit_kuairec_svd(kuairec_matrix, d=d)

    # Persist artifacts
    item_emb_path = artifacts_dir / f"item_embeddings_d{d}.npy"
    user_emb_path = artifacts_dir / f"user_embeddings_d{d}.npy"
    kuai_emb_path = artifacts_dir / f"kuairec_embeddings_d{d}.npy"

    np.save(item_emb_path, e_norm)
    np.save(user_emb_path, u_emb)
    np.save(kuai_emb_path, kuairec_embs)

    logger.info(f"Saved MovieLens item embeddings to {item_emb_path} (shape: {e_norm.shape}).")
    logger.info(f"Saved MovieLens user embeddings to {user_emb_path} (shape: {u_emb.shape}).")
    logger.info(f"Saved KuaiRec item embeddings to {kuai_emb_path} (shape: {kuairec_embs.shape}).")

    return {
        "user_embeddings_shape": u_emb.shape,
        "item_embeddings_shape": e_norm.shape,
        "kuairec_embeddings_shape": kuairec_embs.shape,
        "validation_rmse": float(rmse),
        "num_users": len(user_ids),
        "num_items": len(item_ids),
        "d": d,
    }
