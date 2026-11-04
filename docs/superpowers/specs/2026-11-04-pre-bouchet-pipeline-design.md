# Design Specification: Pre-Bouchet Pipeline & Baseline Infrastructure (Milestone #1)

**Date:** November 4, 2026  
**Project:** Robust Reinforcement Learning for Recommender Systems via *In Silico* LLM Personas (Yale SDS 6190)  
**Author:** Brandon Tran (`brandon.tran@yale.edu`)  
**Status:** Approved by User (Ready for Implementation Planning)

---

## 1. Executive Summary & Objective

This specification details the software architecture, mathematical contracts, and modular execution buckets for the local pre-Bouchet pipeline for Milestone #1 (due November 10, 2026). 

**Timeline Note (Addressing Gap 10):** Following extensive proposal refinement and peer review integration during the second half of October, Phase 1 (Fidelity & HPC setup) and Phase 2 (Classical Baselines) are executed concurrently within an accelerated 6-day sprint (Nov 4 – Nov 10) leading into the Milestone #1 deliverable.

The primary design principle is **asymmetric compute decoupling**:
1. **Local Machine (Windows / Laptop):** Executes all data ingestion, SVD matrix factorization, behavioral signature extraction ($\bm{\phi}(u)$), $k$-means archetype clustering, the statistical RecSim environment ($\mathcal{M}_{\text{stat}}$), the continuous Soft Actor-Critic (SAC) agent, and the offline Conservative Q-Learning (CQL) baseline in pure PyTorch.
2. **GPU Cluster (Yale Bouchet HPC):** Serves `Meta-Llama-3-8B-Instruct` via `vLLM` with Automatic Prefix Caching (APC) to execute the **3-way personalization ablation suite** ($\psi_u$ vs. $\psi_{\text{generic}}$ vs. $\psi_\emptyset$) and compute pre-RL persona fidelity metrics.

All local operations are organized into **atomic, checkpointed execution buckets** running in under 3 minutes each, ensuring that laptop closures or sleep events cause zero data loss.

---

## 2. Directory Layout & Module Structure

```
SDS 6190 - Sequential Decision Making/
├── pyproject.toml                       # Python package configuration (pip install -e .)
├── artifacts/                           # Precomputed representations & checkpoints
│   ├── item_embeddings_d32.npy          # Unit-normalized MovieLens SVD embeddings (|I| x 32)
│   ├── item_metadata.json               # Movie ID -> Title, Year, Genres, Rich Synopsis
│   ├── user_archetypes_k32.json         # 32 pinned archetypes (phi(u), prompt, tau_quit)
│   ├── data_splits.pkl                  # MovieLens D_init (60%) and D_eval (40%)
│   ├── kuairec_matrix.npy               # KuaiRec fully-observed dense rating matrix (1411 x 3327)
│   ├── kuairec_embeddings_d32.npy       # KuaiRec SVD item embeddings (|I_kuai| x 32)
│   ├── sac_recsim_baseline.pt           # Trained SAC Actor & Critic state dicts
│   ├── cql_baseline.pt                  # Trained offline CQL state dict
│   └── fidelity_ablation_results.json   # Pre-RL benchmark results from Bouchet HPC
├── data/                                # Raw datasets (gitignored)
│   ├── ml-1m/                           # MovieLens-1M raw .dat files
│   └── kuairec/                         # KuaiRec small_matrix.csv
├── docs/
│   └── superpowers/specs/               # Design specifications & reviews
│       ├── 2026-11-04-pre-bouchet-pipeline-design.md
│       └── 2026-11-04-design-spec-review.md
├── scripts/                             # Atomic numbered execution runners
│   ├── 01_fetch_and_split_data.py       # [LOCAL] Bucket 1: Download & 60/40 splits (ML-1M + KuaiRec)
│   ├── 02_compute_embeddings_and_archetypes.py # [LOCAL] Bucket 2: SVD, phi(u), k-means
│   ├── 03_test_recsim_env.py            # [LOCAL] Bucket 3: Unit test Gymnasium env
│   ├── 04_train_recsim_sac.py           # [LOCAL] Bucket 4: Train SAC in M_stat
│   ├── 05_train_offline_cql.py          # [LOCAL] Bucket 5: Train CQL on static logs
│   ├── 06_prep_bouchet_payload.py       # [LOCAL] Bucket 6: Package HPC payload
│   └── 07_plot_milestone_figures.py     # [LOCAL] Bucket 7: Generate figures for writeup
├── slurm/                               # HPC Slurm batch submission scripts
│   ├── 01_launch_vllm.sh                # Slurm: Boot vLLM with Prefix Caching
│   └── 02_run_fidelity_ablation.sh      # Slurm: Execute 3-way ablation batch run
└── src/                                 # Pure Python core library (with __init__.py files)
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                    # Dataset downloads, splits, & rich synopsis generator
    │   ├── embeddings.py                # Truncated SVD (d=32) factorizer with RMSE sanity check
    │   └── archetypes.py                # phi(u) signatures & k-means clustering
    ├── envs/
    │   ├── __init__.py
    │   ├── base_env.py                  # Core Gymnasium RecSys POMDP & Boltzmann OPE policy
    │   ├── recsim_stat_env.py           # M_stat with decoupled latent + genre-space satiation
    │   └── mock_llm_env.py              # Lightweight local emulator for vLLM API
    ├── models/
    │   ├── __init__.py
    │   ├── sac.py                       # Pure PyTorch [256, 256] SAC Actor & Twin Critics
    │   └── cql.py                       # Pure PyTorch Conservative Q-Learning
    └── metrics/
        ├── __init__.py
        ├── fidelity.py                  # D_JS, W_1, RMSE, NDCG@10, CI(u)
        └── ope.py                       # Doubly Robust & SNIPS off-policy estimators
```

---

## 3. Detailed Component Specifications

### 3.1 Bucket 1: Data Ingestion & Chronological Partitioning (`src/data/loader.py`)
* **Sources & Roles:**
  * **MovieLens-1M:** Automatically downloaded from `https://files.grouplens.org/datasets/movielens/ml-1m.zip` if not present in `data/ml-1m/`. Contains $1,000,209$ anonymous ratings from $6,040$ users across $3,706$ movies. Serves as the primary training ground for $\mathcal{M}_{\text{stat}}$, $\mathcal{M}_{\text{LLM}}$, SVD embeddings, and offline CQL logs.
  * **KuaiRec (Resolving Gap 2):** Downloaded from official KuaiRec release (`small_matrix.csv`). Contains $1,411$ users and $3,327$ items with $99.6\%$ full observation. **Role:** Serves strictly as a **zero-shot counterfactual evaluation benchmark** (Option A). Because $99.6\%$ of user-item pairs are observed, it provides an uncorrupted ground-truth test bed for off-policy evaluation without exposure bias or missing-data imputation.
* **Filtering & Splitting:**
  * For MovieLens-1M, users with fewer than $50$ ratings are filtered out.
  * For each user $u$, ratings are sorted chronologically by timestamp $t$. The first $60\%$ of ratings form the training history $\mathcal{D}_{\text{init}}$, and the remaining $40\%$ form the held-out evaluation trajectory $\mathcal{D}_{\text{eval}}$.
* **Metadata Extraction & Enriched Synopsis Generator (Resolving Gap 8):**
  * Movie title and release year are parsed: e.g. `"Heat (1995)"` $\to$ title: `"Heat"`, year: `1995`.
  * Genres are extracted as a list from the 18 MovieLens genres.
  * To avoid generic, repetitive strings and ensure genuine semantic prompt conditioning, `loader.py` implements an enriched multi-genre template generator. It dynamically synthesizes narrative premises using the release decade, lead genre tropes, secondary themes, and stylistic modifiers (e.g. *"A gritty 1990s crime drama exploring themes of suspense, cat-and-mouse surveillance, and loyalty."*). Cached to `artifacts/item_metadata.json`.
* **Runtime:** $\sim 20$ seconds on CPU. Output: `artifacts/data_splits.pkl`, `artifacts/item_metadata.json`, and `artifacts/kuairec_matrix.npy`.

---

### 3.2 Bucket 2: Latent Embeddings & Behavioral Grounding (`src/data/embeddings.py`, `src/data/archetypes.py`)

#### 1. SVD Latent Embeddings & RMSE Sanity Check (Resolving Gap 7)
* Build sparse user-item rating matrix $\mathbf{R}_{\text{init}} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ from MovieLens $\mathcal{D}_{\text{init}}$.
* Perform Truncated SVD ($d=32$) using `scipy.sparse.linalg.svds`:
  $$\mathbf{R}_{\text{init}} \approx \mathbf{U} \bm{\Sigma} \mathbf{V}^\top \implies \bm{u}_{\text{emb}} = \mathbf{U} \bm{\Sigma}^{1/2} \in \mathbb{R}^{32}, \quad \bm{e}_i = \mathbf{V} \bm{\Sigma}^{1/2} \in \mathbb{R}^{32}$$
* **RMSE Validation Assertion:** Compute reconstructed rating RMSE against a 10% held-out validation slice of $\mathcal{D}_{\text{init}}$. Verify and log that $\text{RMSE} \in [0.80, 0.95]$ (matching proposal benchmark $0.874$).
* Unit-normalize item vectors: $\hat{\bm{e}}_i = \bm{e}_i / \|\bm{e}_i\|_2$ to enable exact Cosine Similarity MIPS via inner products.
* **KuaiRec SVD Item Embeddings (Resolving Gap 2):** Fit an independent $d=32$ Truncated SVD on KuaiRec's dense matrix $\mathbf{R}_{\text{kuai}} \in \mathbb{R}^{1411 \times 3327}$ to obtain unit-normalized KuaiRec item representations $\hat{\bm{e}}_j^{\text{kuai}} \in \mathbb{R}^{32}$, saved to `artifacts/kuairec_embeddings_d32.npy`. Continuous actions $\bm{a}_t \in [-1, 1]^{32}$ from evaluated policies map directly to KuaiRec items via cosine similarity, querying ground truth from `kuairec_matrix.npy`.

#### 2. Quantitative Behavioral Signatures $\bm{\phi}(u)$
Every user trajectory maps to a 4-component signature $\bm{\phi}(u) = [\bm{\mu}_u, H_u, \lambda_u, \rho_u]^\top$:
1. **Genre Simplex $\bm{\mu}_u \in \Delta^{17}$:** Rating-weighted empirical distribution across $G=18$ genres:
   $$\mu_{u, g} = \frac{\sum_{t \in \mathcal{D}_u} r_t \cdot \mathbb{I}(g \in \text{genres}(i_t))}{\sum_{g'=1}^G \sum_{t \in \mathcal{D}_u} r_t \cdot \mathbb{I}(g' \in \text{genres}(i_t))}$$
2. **Exploration Entropy $H_u \in [0, \log 18]$:** Shannon entropy of the genre distribution:
   $$H_u = -\sum_{g=1}^{18} \mu_{u, g} \log(\mu_{u, g} + 10^{-12})$$
3. **Satiation Decay Slope $\lambda_u \in \mathbb{R}$:** Ordinary Least Squares (OLS) slope of rating vs. repetition count when the user was exposed to the same primary genre in a 3-step window:
   $$\lambda_u = \frac{\sum (k - \bar{k})(r - \bar{r})}{\sum (k - \bar{k})^2}$$
4. **Popularity Conformity $\rho_u \in [-1, 1]$:** Spearman rank correlation between user ratings $r_{u, i}$ and the global item popularity $\log(1 + N_i)$.

#### 3. $k$-Means Archetype Clustering ($k=32$)
* Concatenate $[\bm{u}_{\text{emb}}, \bm{\phi}(u)] \in \mathbb{R}^{32 + 21} = \mathbb{R}^{53}$.
* Fit $k$-means clustering with $k=32$ clusters.
* For each centroid $c_k$, select the closest empirical user $u_k^*$ as the exemplar archetype.
* Determine personalized quit threshold $\tau_{\text{quit}}(u_k^*)$ as the 15th percentile of their historical ratings in $\mathcal{D}_{\text{init}}$.
* Format Llama-3-8B persona system prompt $\psi_{u_k^*}$ summarizing top genres, favorite movies, satiation sensitivity, and rating persona.
* **Runtime:** $\sim 30$ seconds on CPU. Output: `artifacts/item_embeddings_d32.npy`, `artifacts/kuairec_embeddings_d32.npy`, and `artifacts/user_archetypes_k32.json`.

---

### 3.3 Bucket 3: Gymnasium Simulation Environment $\mathcal{M}_{\text{stat}}$ (`src/envs/base_env.py`, `src/envs/recsim_stat_env.py`)

* **POMDP State & Observation Space $\mathcal{O} \in \mathbb{R}^{202}$ (Resolving Gap 3):**
  $$o_t = [\bm{u}_{\text{emb}}, \bm{e}_{i_{t-1}}, r_{t-1}, \Delta t_1, \dots, \bm{e}_{i_{t-5}}, r_{t-5}, \Delta t_5]^\top \in \mathbb{R}^{32 + 5 \times (32 + 1 + 1)} = \mathbb{R}^{202}$$
  Zero-padded at $t=0$. The temporal feature $\Delta t_k$ is formally defined as the **normalized step recency lag**:
  $$\Delta t_k = \frac{k}{K} \in \{0.2, 0.4, 0.6, 0.8, 1.0\} \quad (\text{for } k \in \{1, \dots, 5\})$$
  providing scale-invariant, deterministic temporal decay across both $\mathcal{M}_{\text{stat}}$ and $\mathcal{M}_{\text{LLM}}$.
* **Continuous Action Space & Boltzmann OPE Policy (Resolving Gap 1):**
  * In rollouts, continuous action $\bm{a}_t \in [-1, 1]^{32}$ maps deterministically to an unconsumed item via Cosine MIPS:
    $$i_t = \arg\max_{i \in \mathcal{I} \setminus \mathcal{I}_{u, <t}} \frac{\bm{a}_t^\top \hat{\bm{e}}_i}{\|\bm{a}_t\|_2}$$
  * For Off-Policy Evaluation (OPE), `base_env.py` and `src/metrics/ope.py` implement `get_action_distribution(a_t, unconsumed_mask, tau_pi=0.1)` which computes the full Boltzmann distribution over eligible items:
    $$\pi_e(i \mid o_t) = \frac{\exp\left(\frac{\bm{a}_t^\top \hat{\bm{e}}_i}{\tau_\pi \|\bm{a}_t\|_2}\right)}{\sum_{j \in \mathcal{I} \setminus \mathcal{I}_{u, <t}} \exp\left(\frac{\bm{a}_t^\top \hat{\bm{e}}_j}{\tau_\pi \|\bm{a}_t\|_2}\right)}$$
    This provides explicit, strictly positive propensities $\pi_e(i_t \mid o_t)$ required by Doubly Robust (DR) and Self-Normalized Importance Sampling (SNIPS) estimators.
* **Decoupled Latent and Genre-Space Satiation Dynamics in $\mathcal{M}_{\text{stat}}$ (Resolving Gap 5):**
  * To fix the dimension mismatch between $\mathbb{R}^{32}$ latent space and 18 genres, $\mathcal{M}_{\text{stat}}$ cleanly decouples collaborative affinity from topic satiation:
    1. **Genre Satiation Vector $\bm{w}_t \in \mathbb{R}^{18}$:** Initialized to the archetype's empirical genre simplex $\bm{\mu}_u \in \Delta^{17}$. When item $i_t$ with multi-hot genre vector $\bm{g}_{i_t} \in \{0, 1\}^{18}$ is consumed, genre weights decay exponentially:
       $$w_{t+1, g} = w_{t, g} \cdot \exp(-\beta \cdot g_{i_t, g}) \quad (\text{decay rate } \beta = 0.3)$$
    2. **Latent Preference $\bm{u}_{\text{emb}} \in \mathbb{R}^{32}$:** Remains static, capturing baseline taste.
    3. **Composite Expected Rating:**
       $$\bar{r}_t = 1 + 4 \cdot \sigma\left( \bm{u}_{\text{emb}}^\top \hat{\bm{e}}_{i_t} + \lambda_{\text{satiate}} \cdot (\bm{w}_t^\top \bm{g}_{i_t} - \bar{w}_0) \right)$$
       where $\lambda_{\text{satiate}} = 1.5$ and $\sigma$ is the sigmoid function, with Gaussian noise $\epsilon_t \sim \mathcal{N}(0, 0.2^2)$ clamped to $[1, 5]$.
* **Endogenous Session Continuation & Reward:**
  * Retention flag $c_t = 1 \iff \bar{r}_t \ge \tau_{\text{quit}}(u)$ and consecutive low ratings ($r_t \le 2$) $< 2$.
  * Normalized composite reward ($\alpha = 0.25$):
    $$R(o_t, \bm{a}_t) = \frac{r_t - 3}{2} + 0.25 \cdot \mathbb{I}(c_t = 1) \in [-1.0, 1.25]$$
  * Episode terminates when $c_t = 0$ or step horizon $H=20$ is reached.
* **Discount Factor (Resolving Gap 4):** Formally set to $\gamma = 0.99$.
* **Runtime:** $>2,000$ steps/sec on CPU. Verified via `scripts/03_test_recsim_env.py`.

---

### 3.4 Bucket 4: Soft Actor-Critic (SAC) Implementation (`src/models/sac.py`, `scripts/04_train_recsim_sac.py`)

* **Actor Network ($\pi_\theta$):**
  * Input: $o_t \in \mathbb{R}^{202}$.
  * Architecture: `Linear(202, 256)` $\to$ `LayerNorm` $\to$ `ReLU` $\to$ `Linear(256, 256)` $\to$ `LayerNorm` $\to$ `ReLU` $\to$ `Linear(256, 64)`.
  * Outputs mean $\bm{\mu}_\theta(o_t) \in \mathbb{R}^{32}$ and $\log \bm{\sigma}_\theta(o_t) \in \mathbb{R}^{32}$ (clamped to $[-20, 2]$).
  * Sample $\tilde{\bm{a}} \sim \mathcal{N}(\bm{\mu}, \bm{\sigma}^2)$, squashed via $\bm{a} = \tanh(\tilde{\bm{a}})$.
* **Twin Critic Networks ($Q_{\phi_1}, Q_{\phi_2}$):**
  * Input: Concatenated $(o_t, \bm{a}_t) \in \mathbb{R}^{234}$.
  * Architecture: `Linear(234, 256)` $\to$ `LayerNorm` $\to$ `ReLU` $\to$ `Linear(256, 256)` $\to$ `LayerNorm` $\to$ `ReLU` $\to$ `Linear(256, 1)`.
  * Target networks updated via Polyak averaging: $\phi_{\text{targ}} \leftarrow 0.005 \phi + 0.995 \phi_{\text{targ}}$ with discount factor $\gamma = 0.99$ (Resolving Gap 4):
    $$y_t = R_t + \gamma (1 - d_t) \left( \min_{j=1,2} Q_{\bar{\phi}_j}(o_{t+1}, \tilde{\bm{a}}_{t+1}) - \alpha \log \pi(\tilde{\bm{a}}_{t+1} \mid o_{t+1}) \right)$$
* **Entropy Optimization:**
  * Target entropy $\bar{\mathcal{H}} = -32$. Dynamic temperature $\alpha$ tuned via $\mathcal{L}(\alpha) = -\alpha (\log \pi(\bm{a} \mid o) + \bar{\mathcal{H}})$.
* **Methodological Safeguard Note (Resolving Gap 6):** As specified in Section 4.2 of Proposal v3, if continuous SAC encounters critic divergence across large catalogs during Phase 3, the architecture supports a deferred discrete fallback: candidate items are pre-filtered to the top $M=50$ via SVD inner product, and action selection switches to a Gumbel-Softmax discrete policy head over the top-50 logits.
* **Training Hyperparameters:**
  * Total environment steps: $50,000$. Batch size: $256$. Learning rate: $3 \times 10^{-4}$ (AdamW). Replay buffer size: $100,000$. Discount factor: $\gamma = 0.99$.
  * Runtime on CPU: $\sim 2.5$ minutes. Output: `artifacts/sac_recsim_baseline.pt` and `artifacts/sac_recsim_curve.png`.

---

### 3.5 Bucket 5: Offline Conservative Q-Learning (CQL) Baseline & OPE Contracts (`src/models/cql.py`, `src/metrics/ope.py`, `scripts/05_train_offline_cql.py`)

* **Dataset Construction:**
  * Translates historical interaction sequences in $\mathcal{D}_{\text{init}}$ into offline transitions $(o_t, \bm{a}_t^{\text{log}}, r_t, o_{t+1}, c_t)$, where $\bm{a}_t^{\text{log}} = \hat{\bm{e}}_{i_t} \in \mathbb{R}^{32}$.
* **Objective:**
  $$\min_Q \alpha_{\text{CQL}} \left( \mathbb{E}_{o \sim \mathcal{D}, \bm{a} \sim \text{Uniform}[-1, 1]^{32}}[Q(o, \bm{a})] - \mathbb{E}_{(o, \bm{a}) \sim \mathcal{D}}[Q(o, \bm{a})] \right) + \frac{1}{2} \mathbb{E}_{(o, \bm{a}, r, o') \sim \mathcal{D}}\left[ (Q(o, \bm{a}) - y)^2 \right]$$
  with conservative weight $\alpha_{\text{CQL}} = 1.0$ and $\gamma = 0.99$.
* **OPE Module Contracts (`src/metrics/ope.py` - Resolving Gap 1):**
  * Implements **Doubly Robust (DR)** and **Self-Normalized Importance Sampling (SNIPS)** estimators:
    $$\hat{V}_{\text{DR}}(\pi_e) = \sum_{t=0}^{H} \gamma^t \left( \hat{Q}(o_t, a_t^{\pi_e}) + \frac{\pi_e(a_t \mid o_t)}{\pi_b(a_t \mid o_t)} \left( r_t - \hat{Q}(o_t, a_t) \right) \right)$$
  * Evaluates policy value on held-out MovieLens trajectories $\mathcal{D}_{\text{eval}}$ and zero-shot on the fully-observed KuaiRec matrix.
* **Runtime:** $\sim 3$ minutes on CPU. Output: `artifacts/cql_baseline.pt`.

---

### 3.6 Bucket 6: HPC Bridge & Pre-Bouchet Verification (`slurm/`, `src/envs/mock_llm_env.py`)

* **`src/envs/mock_llm_env.py`:** Emulates the exact `vLLM` HTTP API interface locally using cosine distance and Dirichlet noise, allowing full verification of rollout loops before HPC dispatch.
* **`slurm/01_launch_vllm.sh`:**
  * Requests 1 GPU (L40S / A100), 8 CPUs, 32GB RAM on Bouchet `gpu` partition.
  * Runs: `vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --enable-prefix-caching --gpu-memory-utilization 0.90`.
* **`slurm/02_run_fidelity_ablation.sh`:**
  * Evaluates held-out ratings in $\mathcal{D}_{\text{eval}}$ across:
    1. $\psi_u$: Full historical persona.
    2. $\psi_{\text{generic}}$: Neutral instruction (*"You are an average viewer"*).
    3. $\psi_\emptyset$: Unconditioned candidate metadata only.
  * Outputs: RMSE, NDCG@10, $D_{\text{JS}}$, $W_1$, $\text{CI}(u)$ into `artifacts/fidelity_ablation_results.json`.
* **`scripts/07_plot_milestone_figures.py`:** Generates all 3 publication figures and LaTeX Table 1 for the Milestone #1 writeup.

---

## 4. Verification & Testing Strategy

1. **Unit Tests (`scripts/03_test_recsim_env.py`):**
   * Observation dimension check: `assert obs.shape == (202,)`.
   * Action validity check: `assert np.all(action >= -1.0) and np.all(action <= 1.0)`.
   * Repetition prevention: Verify set of recommendations has length equal to step count.
   * Churn termination check: Verify session ends early on persistent negative ratings.
2. **Model Loss & Convergence Checks:**
   * SAC Critic loss decreases and Q-values remain stable (no divergence).
   * SAC Entropy smoothly converges to target $\bar{\mathcal{H}} = -32$.
   * CQL Q-values on logged actions remain consistently higher than unobserved actions.

---

## 5. Next Steps

Upon review and approval of this specification, implementation will proceed via the `writing-plans` skill to generate the detailed, step-by-step implementation plan.

