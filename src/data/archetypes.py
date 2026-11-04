"""Behavioral signatures phi(u) and k-means archetype extraction for user personas.

Implements Bucket 2b of the Pre-Bouchet Pipeline:
- Formalizes a 21-dimensional behavioral signature phi(u) = [mu_u, H_u, lambda_u, rho_u]^T
- Clusters k=32 archetypes from concatenated latent [u_emb, phi(u)] in R^53
- Extracts exemplar user histories, personalized quit thresholds tau_quit, and Llama-3 persona prompts.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Canonical alphabetical ordering of 18 MovieLens genres
CANONICAL_GENRES: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

GENRE_TO_IDX: Dict[str, int] = {genre: idx for idx, genre in enumerate(CANONICAL_GENRES)}


def compute_behavioral_signature(
    ratings: np.ndarray,
    genre_matrix: np.ndarray,
    pop_counts: np.ndarray,
    primary_genres: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes the 21-dimensional behavioral signature phi(u) for a user trajectory.

    Signature components:
        1. Genre Simplex mu_u in Delta^17 (18 dimensions):
           Rating-weighted empirical distribution across 18 genres.
        2. Exploration Entropy H_u in [0, log 18] (1 dimension):
           Shannon entropy of the genre distribution.
        3. Satiation Decay Slope lambda_u in R (1 dimension):
           OLS slope of ratings against same-genre repetition in a 3-step window.
        4. Popularity Conformity rho_u in [-1, 1] (1 dimension):
           Spearman rank correlation between ratings and global log-popularity.

    Args:
        ratings: 1D array of user ratings r_t in [1.0, 5.0].
        genre_matrix: 2D array of multi-hot or binary genre indicators (T, 18).
        pop_counts: 1D array of global item interaction counts N_{i_t} (T,).
        primary_genres: Optional 1D array of primary genre indices (T,). If None,
            derived as the argmax genre from genre_matrix.

    Returns:
        np.ndarray of shape (21,) with dtype float32.
    """
    ratings = np.asarray(ratings, dtype=np.float32)
    genre_matrix = np.asarray(genre_matrix, dtype=np.float32)
    pop_counts = np.asarray(pop_counts, dtype=np.float32)

    T = len(ratings)
    if T == 0 or genre_matrix.shape[0] == 0:
        # Default fallback: uniform simplex, zero entropy, zero satiation, zero popularity
        mu = np.full(18, 1.0 / 18.0, dtype=np.float32)
        return np.concatenate([mu, [0.0, 0.0, 0.0]], dtype=np.float32)

    # 1. Genre Simplex mu_u in Delta^17
    # mu_{u, g} = sum_t (r_t * I(g in genres(i_t))) / sum_{g'} sum_t (r_t * I(g' in genres(i_t)))
    weighted_genres = ratings[:, None] * genre_matrix  # (T, 18)
    genre_sums = np.sum(weighted_genres, axis=0)  # (18,)
    total_weight = float(np.sum(genre_sums))

    if total_weight > 0:
        mu = (genre_sums / total_weight).astype(np.float32)
    else:
        mu = np.full(18, 1.0 / 18.0, dtype=np.float32)

    # 2. Exploration Entropy H_u in [0, log 18]
    # H_u = - sum_{g=1}^18 mu_{u, g} * log(mu_{u, g} + eps)
    pos_mask = mu > 0
    if np.any(pos_mask):
        H_u = float(-np.sum(mu[pos_mask] * np.log(mu[pos_mask])))
    else:
        H_u = 0.0
    H_u = float(np.clip(H_u, 0.0, np.log(18.0)))

    # 3. Satiation Decay Slope lambda_u in R
    # Find pairs or sub-sequences within window <= 3 where primary genre matches:
    # lambda_u = sum((k - bar_k)(r - bar_r)) / sum((k - bar_k)^2)
    if primary_genres is None:
        has_genre = np.max(genre_matrix, axis=1) > 0
        primary_genres_arr = np.where(has_genre, np.argmax(genre_matrix, axis=1), -1)
    else:
        primary_genres_arr = np.asarray(primary_genres, dtype=int)

    valid_chains: List[Tuple[np.ndarray, np.ndarray]] = []

    # Identify repetition chains for each primary genre
    for g in range(18):
        occurrences = np.where(primary_genres_arr == g)[0]
        if len(occurrences) < 2:
            continue

        current_chain: List[int] = [occurrences[0]]
        for idx in occurrences[1:]:
            # If within 3 steps of previous occurrence of the same genre
            if idx - current_chain[-1] <= 3:
                current_chain.append(idx)
            else:
                if len(current_chain) >= 2:
                    k_chain = np.arange(len(current_chain), dtype=np.float64)
                    r_chain = np.array([ratings[t_idx] for t_idx in current_chain], dtype=np.float64)
                    valid_chains.append((k_chain, r_chain))
                current_chain = [idx]

        if len(current_chain) >= 2:
            k_chain = np.arange(len(current_chain), dtype=np.float64)
            r_chain = np.array([ratings[t_idx] for t_idx in current_chain], dtype=np.float64)
            valid_chains.append((k_chain, r_chain))

    if valid_chains:
        cov_kr = 0.0
        var_k = 0.0
        for k_chain, r_chain in valid_chains:
            k_centered = k_chain - np.mean(k_chain)
            r_centered = r_chain - np.mean(r_chain)
            cov_kr += float(np.sum(k_centered * r_centered))
            var_k += float(np.sum(k_centered ** 2))

        if var_k > 1e-9:
            lambda_u = float(cov_kr / var_k)
        else:
            lambda_u = 0.0
    else:
        lambda_u = 0.0

    # 4. Popularity Conformity rho_u in [-1, 1]
    # Spearman rank correlation between ratings and log(1 + N_i)
    if T < 2 or np.all(ratings == ratings[0]) or np.all(pop_counts == pop_counts[0]):
        rho_u = 0.0
    else:
        log_pop = np.log1p(pop_counts)
        res = spearmanr(ratings, log_pop)
        stat = getattr(res, "statistic", res[0])
        if np.isnan(stat):
            rho_u = 0.0
        else:
            rho_u = float(np.clip(stat, -1.0, 1.0))

    phi = np.empty(21, dtype=np.float32)
    phi[:18] = mu
    phi[18] = H_u
    phi[19] = lambda_u
    phi[20] = rho_u

    return phi


def compute_all_user_signatures(
    train_interactions: Dict[Any, List[Tuple]],
    item_metadata: Dict[str, Dict],
    user_ids: List[Any],
    item_ids: List[Any],
) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Computes behavioral signatures for all users in the training dataset.

    Args:
        train_interactions: Mapping from user_id to chronological interaction tuples (item_id, rating, ...).
        item_metadata: Mapping from item_id (str or int) to metadata dictionary with 'genres' and 'title'.
        user_ids: Canonical list of user IDs determining output matrix row order.
        item_ids: Canonical list of item IDs.

    Returns:
        tuple containing:
            - user_signatures: np.ndarray of shape (|U|, 21) with float32 values.
            - item_popularity: Dict mapping item_id to global interaction count N_i in train set.
    """
    # Step 1: Compute global item popularity N_i across training interactions
    item_popularity: Dict[Any, int] = {iid: 0 for iid in item_ids}
    for uid, inters in train_interactions.items():
        for inter in inters:
            iid = inter[0]
            if iid in item_popularity:
                item_popularity[iid] += 1
            else:
                item_popularity[iid] = 1

    # Step 2: Precompute multi-hot genre vectors and primary genres for all items
    item_genre_vecs: Dict[Any, np.ndarray] = {}
    item_primary_genres: Dict[Any, int] = {}

    for iid in item_ids:
        meta = item_metadata.get(str(iid)) or item_metadata.get(iid, {})
        genres = meta.get("genres", [])
        g_vec = np.zeros(18, dtype=np.float32)
        for g in genres:
            if g in GENRE_TO_IDX:
                g_vec[GENRE_TO_IDX[g]] = 1.0

        item_genre_vecs[iid] = g_vec
        if genres and genres[0] in GENRE_TO_IDX:
            item_primary_genres[iid] = GENRE_TO_IDX[genres[0]]
        elif np.any(g_vec > 0):
            item_primary_genres[iid] = int(np.argmax(g_vec))
        else:
            item_primary_genres[iid] = -1

    # Step 3: Compute signature for each user
    num_users = len(user_ids)
    user_signatures = np.zeros((num_users, 21), dtype=np.float32)

    for u_idx, uid in enumerate(user_ids):
        inters = train_interactions.get(uid, [])
        if not inters:
            user_signatures[u_idx] = compute_behavioral_signature(
                ratings=np.empty(0, dtype=np.float32),
                genre_matrix=np.empty((0, 18), dtype=np.float32),
                pop_counts=np.empty(0, dtype=np.float32),
            )
            continue

        u_ratings = np.array([float(x[1]) for x in inters], dtype=np.float32)
        u_pop = np.array([float(item_popularity.get(x[0], 0)) for x in inters], dtype=np.float32)
        u_genres = np.stack(
            [item_genre_vecs.get(x[0], np.zeros(18, dtype=np.float32)) for x in inters],
            axis=0,
        )
        u_primary = np.array(
            [item_primary_genres.get(x[0], -1) for x in inters],
            dtype=int,
        )

        user_signatures[u_idx] = compute_behavioral_signature(
            ratings=u_ratings,
            genre_matrix=u_genres,
            pop_counts=u_pop,
            primary_genres=u_primary,
        )

    return user_signatures, item_popularity


def extract_k_archetypes(
    user_embeddings: np.ndarray,
    user_signatures: np.ndarray,
    user_ids: List[Any],
    train_interactions: Dict[Any, List[Tuple]],
    item_metadata: Dict[str, Dict],
    k: int = 32,
    random_state: int = 42,
    normalize_blocks: bool = True,
) -> List[Dict[str, Any]]:
    """Extracts k representative user archetypes via k-means clustering in joint R^53 space.

    Concatenates latent factor embeddings u_emb in R^32 and behavioral signatures phi(u) in R^21
    to form z_u in R^53. Normalizes block variances when normalize_blocks=True to prevent
    signature features from dominating Euclidean distance over latent factors.
    Identifies empirical exemplar users closest to cluster centroids,
    extracts personalized quit thresholds tau_quit, and generates structured persona prompts.

    Args:
        user_embeddings: Latent SVD user embeddings (|U|, 32).
        user_signatures: Behavioral signatures (|U|, 21).
        user_ids: List of canonical user IDs corresponding to row indices.
        train_interactions: Mapping from user_id to interaction history tuples.
        item_metadata: Mapping from item_id to metadata dictionaries with 'title' and 'genres'.
        k: Number of cluster archetypes to extract (default: 32).
        random_state: Random seed for KMeans reproducibility (default: 42).
        normalize_blocks: Whether to scale u_emb and phi(u) to equal standard deviation
            so collaborative and behavioral features contribute balanced variance (default: True).

    Returns:
        List of k archetype dictionaries formatted for simulation and LLM emulation.
    """
    user_embeddings = np.asarray(user_embeddings, dtype=np.float32)
    user_signatures = np.asarray(user_signatures, dtype=np.float32)

    num_users = len(user_ids)
    if user_embeddings.shape[0] != num_users or user_signatures.shape[0] != num_users:
        raise ValueError(
            f"Shape mismatch: {num_users} user_ids vs {user_embeddings.shape[0]} embeddings "
            f"and {user_signatures.shape[0]} signatures."
        )

    # Form joint representation in R^53 with block variance balancing
    if normalize_blocks:
        std_u = float(np.std(user_embeddings))
        scale_u = std_u if std_u > 1e-9 else 1.0
        u_scaled = (user_embeddings / scale_u).astype(np.float32)

        std_phi = float(np.std(user_signatures))
        scale_phi = std_phi if std_phi > 1e-9 else 1.0
        phi_scaled = (user_signatures / scale_phi).astype(np.float32)

        z_clustering = np.hstack([u_scaled, phi_scaled]).astype(np.float32)
    else:
        z_clustering = np.hstack([user_embeddings, user_signatures]).astype(np.float32)

    # Fit k-means clustering
    k_actual = min(k, num_users)
    kmeans = KMeans(n_clusters=k_actual, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(z_clustering)
    centroids = kmeans.cluster_centers_  # (k_actual, 53)

    archetypes: List[Dict[str, Any]] = []
    used_exemplars: set = set()

    for k_idx in range(k_actual):
        centroid = centroids[k_idx]
        # Distances from all users to this centroid in balanced clustering space
        dists = np.linalg.norm(z_clustering - centroid, axis=1)

        # Find closest empirical user, preferring unassigned users
        sorted_indices = np.argsort(dists)
        chosen_idx = None
        for idx in sorted_indices:
            if idx not in used_exemplars:
                chosen_idx = int(idx)
                break
        if chosen_idx is None:
            chosen_idx = int(sorted_indices[0])
        used_exemplars.add(chosen_idx)

        exemplar_uid = user_ids[chosen_idx]
        u_emb_vec = user_embeddings[chosen_idx]
        phi_vec = user_signatures[chosen_idx]
        genre_simplex = phi_vec[:18]
        entropy_val = float(phi_vec[18])
        lambda_val = float(phi_vec[19])
        conformity_val = float(phi_vec[20])

        # Personalized quit threshold tau_quit: 15th percentile of historical ratings, clamped [1.0, 3.5]
        inters = train_interactions.get(exemplar_uid, [])
        if inters:
            u_ratings = [float(x[1]) for x in inters]
            raw_tau = float(np.percentile(u_ratings, 15))
            tau_quit = float(np.clip(raw_tau, 1.0, 3.5))
        else:
            tau_quit = 2.0
        tau_quit = float(round(tau_quit, 2))

        # Top genres by empirical preference
        sorted_g_idx = np.argsort(genre_simplex)[::-1]
        top_genres = [
            CANONICAL_GENRES[idx] for idx in sorted_g_idx[:3] if genre_simplex[idx] > 0
        ]
        if not top_genres:
            top_genres = [CANONICAL_GENRES[sorted_g_idx[0]]]

        # Favorite movies (titles of top rated movies)
        if inters:
            # Sort by rating descending, then timestamp ascending
            sorted_inters = sorted(inters, key=lambda x: (float(x[1]), -float(x[2])), reverse=True)
            favorite_titles: List[str] = []
            for item_tuple in sorted_inters:
                iid = item_tuple[0]
                meta = item_metadata.get(str(iid)) or item_metadata.get(iid, {})
                title = meta.get("title", f"Movie {iid}")
                if title not in favorite_titles:
                    favorite_titles.append(title)
                if len(favorite_titles) >= 5:
                    break
        else:
            favorite_titles = ["Classic Cinema"]

        # Satiation tendency description
        if lambda_val < -0.1:
            satiation_desc = "Gets tired of repeat genres quickly and craves variety."
        elif lambda_val > 0.05:
            satiation_desc = "Loves genre binges and enjoys watching similar movies consecutively."
        else:
            satiation_desc = "Moderate genre patience; tolerates occasional repetitions without strong fatigue."

        # Structured Llama-3 persona prompt
        top_genres_str = ", ".join(top_genres)
        fav_movies_str = ", ".join(favorite_titles[:5])
        system_prompt = (
            f"You are a movie viewer with specific personal tastes.\n"
            f"Favorite Genres: {top_genres_str}\n"
            f"Adored Titles: {fav_movies_str}\n"
            f"Satiation Tendency: {satiation_desc}\n"
            f"Rating Style: You quit watching recommendations when movies fall below rating {tau_quit:.1f}/5.0.\n"
            f"Rate recommended movies realistically on a 1.0 to 5.0 scale reflecting your taste."
        )

        uid_serialized = int(exemplar_uid) if isinstance(exemplar_uid, (int, np.integer)) else exemplar_uid

        archetype_dict: Dict[str, Any] = {
            "archetype_id": int(k_idx),
            "user_id": uid_serialized,
            "u_emb": [float(x) for x in u_emb_vec],
            "phi": [float(x) for x in phi_vec],
            "tau_quit": float(round(tau_quit, 2)),
            "system_prompt": system_prompt,
            "top_genres": top_genres,
            "favorite_movies": favorite_titles[:5],
            "genre_simplex": [float(x) for x in genre_simplex],
        }
        archetypes.append(archetype_dict)

    return archetypes


def run_archetypes_pipeline(
    artifacts_dir: Union[str, Path] = Path("artifacts"),
    d: int = 32,
    k: int = 32,
    random_state: int = 42,
    normalize_blocks: bool = True,
) -> List[Dict[str, Any]]:
    """Runs the complete behavioral signature and archetype extraction pipeline.

    Loads data splits, user embeddings, and item metadata. Computes behavioral signatures
    for all users, extracts k=32 archetypes, and persists `artifacts/user_archetypes_k32.json`.

    Args:
        artifacts_dir: Path to directory containing required artifacts.
        d: SVD embedding dimension (default: 32).
        k: Number of archetypes (default: 32).
        random_state: Random state for k-means (default: 42).
        normalize_blocks: Whether to balance block scale in joint clustering space (default: True).

    Returns:
        List of k archetype dictionaries.
    """
    artifacts_dir = Path(artifacts_dir)
    data_splits_path = artifacts_dir / "data_splits.pkl"
    metadata_path = artifacts_dir / "item_metadata.json"
    user_emb_path = artifacts_dir / f"user_embeddings_d{d}.npy"

    if not data_splits_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {data_splits_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {metadata_path}")
    if not user_emb_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {user_emb_path}")

    logger.info(f"Loading data splits from {data_splits_path}...")
    with open(data_splits_path, "rb") as f:
        splits = pickle.load(f)

    logger.info(f"Loading item metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        item_metadata = json.load(f)

    logger.info(f"Loading user embeddings from {user_emb_path}...")
    user_embs = np.load(user_emb_path)

    user_ids = splits["user_ids"]
    item_ids = splits["item_ids"]
    train_interactions = splits["train_interactions"]

    logger.info(
        f"Computing 21-D behavioral signatures phi(u) for {len(user_ids)} users..."
    )
    user_signatures, item_pop = compute_all_user_signatures(
        train_interactions=train_interactions,
        item_metadata=item_metadata,
        user_ids=user_ids,
        item_ids=item_ids,
    )

    logger.info(
        f"Extracting {k} archetypes via k-means in R^53 (d={d}, signatures=21)..."
    )
    archetypes = extract_k_archetypes(
        user_embeddings=user_embs,
        user_signatures=user_signatures,
        user_ids=user_ids,
        train_interactions=train_interactions,
        item_metadata=item_metadata,
        k=k,
        random_state=random_state,
        normalize_blocks=normalize_blocks,
    )

    # Persist archetypes artifact atomically
    out_path = artifacts_dir / f"user_archetypes_k{k}.json"
    tmp_path = artifacts_dir / f"user_archetypes_k{k}.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(archetypes, f, indent=2)
    tmp_path.replace(out_path)

    logger.info(
        f"Saved {len(archetypes)} user archetypes atomically to {out_path}."
    )

    return archetypes
