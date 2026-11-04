# Design Spec Review: Gaps Against `project_proposal_v3.tex`

**Reviewer:** Claude Opus (requested by user)  
**Date:** November 4, 2026  
**Documents Under Review:**
- Design Spec: [`docs/superpowers/specs/2026-11-04-pre-bouchet-pipeline-design.md`](file:///c:/Users/brand/OneDrive%20-%20Yale%20University/3_Fall%202026/SDS%206190%20-%20Sequential%20Decision%20Making/docs/superpowers/specs/2026-11-04-pre-bouchet-pipeline-design.md)
- Source of Truth: [`Proposal/project_proposal_v3.tex`](file:///c:/Users/brand/OneDrive%20-%20Yale%20University/3_Fall%202026/SDS%206190%20-%20Sequential%20Decision%20Making/Proposal/project_proposal_v3.tex)

---

> [!IMPORTANT]
> Gaps 1–5 should be resolved in the design spec before generating the implementation plan. Gaps 6–10 are low-severity and can be addressed during implementation.

---

## Gap 1 · Boltzmann Softmax Policy for OPE (HIGH)

**Proposal reference:** Line 123 — *"inducing a Boltzmann distribution $\pi_e(i \mid o_t) \propto \exp(\cos(\bm{a}_t, \bm{e}_i)/\tau_{\pi})$ for off-policy evaluation"*

**What's missing:** The design spec's environment (Section 3.3) only implements hard `argmax` cosine similarity for item selection. There is no mechanism to compute the full softmax distribution over unconsumed items, which is required to produce importance weights for Doubly Robust (DR) and SNIPS off-policy evaluation.

The file `src/metrics/ope.py` is listed in the directory tree but its contract is never specified.

**Suggested resolution:** Add a method to `base_env.py` (or to `ope.py`) that, given a continuous action $\bm{a}_t$ and the current unconsumed item set, computes:

$$\pi_e(i \mid o_t) = \frac{\exp(\cos(\bm{a}_t, \hat{\bm{e}}_i) / \tau_\pi)}{\sum_{j \in \mathcal{I} \setminus \mathcal{I}_{u,<t}} \exp(\cos(\bm{a}_t, \hat{\bm{e}}_j) / \tau_\pi)}$$

Specify $\tau_\pi$ (e.g., 0.1) and document the `ope.py` module contract (DR estimator, SNIPS estimator, inputs, outputs).

---

## Gap 2 · KuaiRec Processing Is Undefined (HIGH)

**Proposal reference:** Line 121 — *"KuaiRec (1,411 users, 3,327 items, 99.6% matrix density), which provides a fully-observed interaction ground truth that eliminates counterfactual exposure bias in off-policy evaluation."*  
Line 167 — *"On KuaiRec, we processed the fully-observed 1,411 × 3,327 matrix, ensuring an uncorrupted test bed for zero-shot transfer without missing-data imputation."*

**What's missing:** The design spec mentions downloading `small_matrix.csv` in Bucket 1 but never specifies:
1. Whether an independent $d=32$ SVD is fit on KuaiRec interactions, or whether KuaiRec is used purely as a lookup table for ground-truth ratings.
2. What item representations exist for KuaiRec items (they are short videos, not movies — no genres, no titles, no synopses in the standard release).
3. How KuaiRec integrates with the Gymnasium environment, which is currently designed entirely around MovieLens metadata (genre-based satiation, movie synopses).

**Suggested resolution:** Decide and document one of:
- **(A) Lookup-only evaluation:** KuaiRec is not used as a training environment. Policies trained in $\mathcal{M}_{\text{stat}}$ or $\mathcal{M}_{\text{LLM}}$ on MovieLens are evaluated on KuaiRec by mapping continuous actions to KuaiRec items via a separate KuaiRec SVD embedding, then looking up the ground-truth rating in the fully-observed matrix. This avoids the metadata mismatch problem entirely.
- **(B) Independent environment:** Build a separate KuaiRec environment with its own SVD embeddings, item features, and satiation dynamics. This is significantly more work.

Option (A) aligns better with the proposal's language ("zero-shot transfer," "uncorrupted test bed") and is likely the intended design.

---

## Gap 3 · $\Delta t_k$ in the Observation Vector Is Undefined (MEDIUM)

**Proposal reference:** Line 139 — *"the last $K=5$ interaction tuples $\mathcal{H}_t = \{(\bm{e}_{i_{t-k}}, r_{t-k}, \Delta t_k)\}_{k=1}^5$"*

**What's missing:** Neither document defines what $\Delta t_k$ represents. Candidates:
- Wall-clock inter-rating time (seconds or hours) from MovieLens timestamps.
- Relative step index within the session (i.e., $k$ itself).
- A constant placeholder (e.g., 1.0).

In $\mathcal{M}_{\text{stat}}$, there are no real timestamps, so $\Delta t_k$ would need to be synthesized or set to a constant.

**Suggested resolution:** Define $\Delta t_k = k$ (the recency index, where 1 = most recent) in both environments. This is simple, always available, and gives the agent a notion of how far back each history item is. Document this in Section 3.3 of the design spec. Alternatively, set $\Delta t_k = 1.0$ as a constant placeholder and note that real timestamps can be introduced later for MovieLens evaluation.

---

## Gap 4 · Discount Factor $\gamma$ Is Never Specified (MEDIUM)

**Proposal reference:** Line 136 — *"POMDP defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \Omega, \gamma)$"*  
Line 156 — Simulation Lemma bound depends on $\gamma$ via the $\frac{1}{1-\gamma}$ factor.

**What's missing:** No concrete value for $\gamma$ is given anywhere. The SAC critic target computation:

$$y_t = R_t + \gamma (1 - d_t) \left( \min_{j} Q_{\bar{\phi}_j}(o_{t+1}, \tilde{\bm{a}}_{t+1}) - \alpha \log \pi(\tilde{\bm{a}}_{t+1} \mid o_{t+1}) \right)$$

requires a specific number.

**Suggested resolution:** Set $\gamma = 0.99$. With $H=20$ episode horizon, $\gamma^{20} \approx 0.82$, so rewards at the end of the session still carry meaningful weight. Add this to Section 3.4 of the design spec under Training Hyperparameters, and add it to the `base_env.py` configuration.

---

## Gap 5 · Satiation Decay Operates on Wrong Vector Space (MEDIUM)

**Proposal reference:** Line 140 — *"In $\mathcal{M}_{\text{stat}}$ (RecSim), transitions follow a hard-coded parametric decay rule where genre affinity drops exponentially upon repeated exposure."*

**What's missing:** The design spec (Section 3.3) defines:
- Latent user state $\bm{\theta}_t \in \mathbb{R}^{32}$ (initialized to $\bm{u}_{\text{emb}}$, the SVD latent space)
- Decay indexed by genre $g$: $\theta_{t+1,g} = \theta_{t,g} \cdot \exp(-\beta \cdot \mathbb{I}(g \in \text{genres}(i_t)))$

But $g$ indexes 18 genres while $\bm{\theta}_t$ has 32 dimensions. These are different spaces. The decay formula as written treats the first 18 dimensions of a 32-dimensional vector as genre slots, which doesn't match SVD latent semantics.

**Suggested resolution:** Maintain a separate genre affinity vector $\bm{w}_t \in \mathbb{R}^{18}$ initialized to the archetype's genre simplex $\bm{\mu}_u$. Apply the exponential decay to $\bm{w}_t$. Compute expected rating as:

$$\bar{r}_t = 1 + 4 \cdot \sigma\left(\bm{u}_{\text{emb}}^\top \hat{\bm{e}}_{i_t} + \bm{w}_t^\top \bm{g}_{i_t}\right)$$

where $\bm{g}_{i_t} \in \{0,1\}^{18}$ is the item's multi-hot genre vector. This cleanly separates the latent-space item similarity from the genre-space satiation dynamics and makes the parametric exploitability of $\mathcal{M}_{\text{stat}}$ more obvious (which is the point — the SAC agent should overfit to this simple decay).

---

## Gap 6 · Gumbel-Softmax Fallback Safeguard Omitted (LOW)

**Proposal reference:** Line 202 — *"If continuous SAC encounters critic divergence across large catalogs, we pre-filter $M=50$ candidates via SVD inner products and execute discrete Gumbel-Softmax updates."*

**What's missing:** The design spec does not mention this fallback.

**Suggested resolution:** Add a note to Section 3.4 documenting it as a deferred safeguard: if Q-value divergence is detected during SAC training (e.g., critic loss exceeds a threshold for N consecutive updates), fall back to pre-filtering the top $M=50$ items by SVD inner product and switching to Gumbel-Softmax discrete action selection. Implementation is deferred to Phase 3 (LLM-RL) when catalog-scale divergence is more likely.

---

## Gap 7 · SVD Validation RMSE Sanity Check Missing (LOW)

**Proposal reference:** Line 167 — *"trained a $d=32$ regularized SVD model on $\mathcal{D}^{\text{init}}$ (validation RMSE $= 0.874$)"*

**What's missing:** Bucket 2 does not include a validation step to confirm the SVD RMSE matches the expected ballpark.

**Suggested resolution:** After fitting SVD in `src/data/embeddings.py`, compute RMSE on a held-out validation fold (e.g., last 10% of $\mathcal{D}_{\text{init}}$). Print it and assert it's in the range $[0.80, 0.95]$. Log the actual value for the milestone writeup.

---

## Gap 8 · Synthetic Synopsis Quality May Be Insufficient (LOW)

**Proposal reference:** Line 146 — Candidates are presented as `[Title: {title} | Year: {year} | Genres: {genres} | Synopsis: {synopsis}]`.

**What's missing:** The design spec replaces real TMDb synopses with a deterministic offline generator from Title + Decade + Genres. If the output is a mechanical string like *"A 1990s Action Crime film"*, it adds almost no semantic information beyond the genre tags themselves, undermining the proposal's claim that semantic prompt encoding *"forces the model to evaluate thematic and narrative alignment rather than relying exclusively on memorized global popularity."*

**Suggested resolution:** During implementation, spot-check 10–20 generated synopses for quality and diversity. If they're too generic, consider either (a) crafting slightly richer templates per genre combination, or (b) downloading a prebuilt TMDb synopsis dump (several community datasets exist on Kaggle/HuggingFace that map MovieLens IDs to TMDb plot summaries).

---

## Gap 9 · No Python Package Structure (LOW)

**What's missing:** The `src/` directory is intended as an importable library, but the design spec doesn't mention `__init__.py` files, a `pyproject.toml`, or `PYTHONPATH` configuration. On Bouchet HPC, importing `from src.envs.base_env import RecEnv` will fail unless the package is installed or the path is set.

**Suggested resolution:** Add `__init__.py` files to `src/`, `src/data/`, `src/envs/`, `src/models/`, and `src/metrics/`. Add a minimal `pyproject.toml` at the project root with `pip install -e .` support. Alternatively, have each script do `sys.path.insert(0, PROJECT_ROOT)` at the top.

---

## Gap 10 · Timeline Shift Not Acknowledged (LOW)

**Proposal reference:** Lines 190–191 — Phase 1 (Fidelity) was scheduled for Oct 13–26, Phase 2 (Baselines) for Oct 27–Nov 10.

**What's missing:** It is now November 4 and no implementation work has been done. The milestone writeup should acknowledge the shifted timeline.

**Suggested resolution:** In the Milestone 1 report, note that proposal preparation and peer review revision consumed the Phase 1 window, and that the fidelity evaluation + baseline training are being executed concurrently during the Phase 2 window. This is a writeup concern, not a design spec concern.
