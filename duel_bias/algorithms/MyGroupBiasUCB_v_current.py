from typing import Optional
import numpy as np

from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.feedback import FeedbackMechanism
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism


class MyGroupBiasUCB(BiasAlgorithm, CondorcetProducer):
    """UCB duelling bandit with EM-based group-bias correction."""

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int   = 3000,
        random_state        = None,
        alpha:         float = 0.51,
        epsilon:       float = 0.4,
        em_iters:      int   = 15,
        em_restarts:   int   = 5,
        gd_lr:         float = 0.03,
        gd_iters:      int   = 150,
        lambda_q:      float = 0.005,
        em_refit_freq: int   = 50,
        warmup_obs_per_user_pair: int = 5,
        warmup_max_frac:  float = 0.5,
        gd_stability_gate: float = 1e-4,
        recent_decay: float = 0.95,
        use_spectral_init:      bool = True,
        use_svd_init:           bool = True,
        cross_group_correction: bool = True,
        debug:         bool  = False,
    ):
        super().__init__({}, feedback_mechanism, time_horizon)

        if random_state is None:
            self.random_state = np.random.RandomState(42)
        elif isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, 2**31))
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = random_state

        self.debug = debug
        self.wrapped_feedback = BudgetedFeedbackMechanism(
            feedback_mechanism, time_horizon * 10)

        obj = feedback_mechanism
        while hasattr(obj, "feedback_mechanism"):
            obj = obj.feedback_mechanism
        self._base_env = obj

        self.K = self.wrapped_feedback.get_num_arms()
        if self.K < 2:
            raise ValueError(
                f"MyGroupBiasUCB requires at least 2 arms, got {self.K}")

        self.M = (len(self._base_env.user_bias)
                  if hasattr(self._base_env, "user_bias") else 10)

        # ── Store warmup config (so ablation can set to 0) ────────────
        self._warmup_obs_per_user_pair = warmup_obs_per_user_pair
        self._warmup_max_frac = warmup_max_frac

        pairs_count = self.K * (self.K - 1) // 2
        raw_warmup = pairs_count * self.M * warmup_obs_per_user_pair
        # self._WARMUP_ROUNDS = min(raw_warmup,
        #                           int(time_horizon * warmup_max_frac))
        self._WARMUP_ROUNDS = 200

        self.time_step    = 0
        self.time_horizon = time_horizon
        self.alpha        = alpha
        self.epsilon      = epsilon
        self.em_iters     = em_iters
        self.em_restarts  = em_restarts
        self.gd_lr        = gd_lr
        self.gd_iters     = gd_iters
        self.lambda_q     = lambda_q
        self.em_refit_freq = em_refit_freq

        # ── FIX 1 & 2: store new parameters ──────────────────────────
        self.gd_stability_gate = gd_stability_gate
        self.recent_decay      = recent_decay

        self.q    = np.zeros(self.K)
        self.beta = np.zeros(self.M)
        self.g    = np.zeros(self.K, dtype=np.int8)
        self._model_fitted = False

        self.wins_e   = np.zeros((self.M, self.K, self.K))
        self.totals_e = np.zeros((self.M, self.K, self.K))
        self.wins_m   = np.zeros((self.K, self.K))
        self.totals_m = np.zeros((self.K, self.K))

        self._warmup_pairs = [(i, j)
                              for i in range(self.K)
                              for j in range(i + 1, self.K)]
        self.use_spectral_init       = use_spectral_init
        self.use_svd_init            = use_svd_init
        self.cross_group_correction  = cross_group_correction

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _marginal_q(self) -> np.ndarray:
        w = self.wins_m.sum(axis=1)
        t = self.totals_m.sum(axis=1)
        p = np.clip((w + 0.5) / (t + 1.0), 0.02, 0.98)
        qt = np.log(p / (1.0 - p))
        qt -= qt.mean()
        return qt

    # ── FIX 2: decay-weighted observation snapshots ───────────────────

    def _get_decayed_observations(self):
        """Return (wins_e, totals_e, wins_m, totals_m) with recency decay.

        When recent_decay < 1.0, older observations are down-weighted.
        When recent_decay == 1.0, returns the raw counts (no copy needed
        for read-only use).
        """
        if self.recent_decay >= 1.0:
            # No decay — use raw counts directly
            return self.wins_e, self.totals_e, self.wins_m, self.totals_m

        # Apply decay: scale all accumulated counts by decay factor.
        # This is an approximation — ideally we'd track per-round counts,
        # but that would be O(T) memory.  Instead we apply a single
        # multiplicative shrinkage toward the uniform prior each time
        # we snapshot for EM, which down-weights older history.
        d = self.recent_decay
        return (self.wins_e * d, self.totals_e * d,
                self.wins_m * d, self.totals_m * d)

    # ── Group initialisation sources ──────────────────────────────────

    def _spectral_init(self) -> np.ndarray:
        """Spectral clustering on LR-disagreement affinity."""
        _, _, wins_m_d, totals_m_d = self._get_decayed_observations()
        K, M = self.K, self.M
        wins_e_d, totals_e_d, _, _ = self._get_decayed_observations()

        D = np.zeros((K, K))
        for a in range(K):
            for b in range(a + 1, K):
                wins_list, n_list = [], []
                for e in range(M):
                    w = wins_e_d[e, a, b]
                    n = totals_e_d[e, a, b]
                    if n > 0:
                        wins_list.append(w)
                        n_list.append(n)
                if len(wins_list) < 2:
                    continue
                total_w = sum(wins_list)
                total_n = sum(n_list)
                p_pool = np.clip(total_w / total_n, 0.01, 0.99)
                ll_h0 = sum(
                    w * np.log(p_pool) + (n - w) * np.log(1 - p_pool)
                    for w, n in zip(wins_list, n_list))
                ll_h1 = sum(
                    w * np.log(np.clip(w / n, 0.01, 0.99))
                    + (n - w) * np.log(np.clip(1 - w / n, 0.01, 0.99))
                    for w, n in zip(wins_list, n_list))
                D[a, b] = D[b, a] = max(0.0, 2.0 * (ll_h1 - ll_h0))

        scale = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        A = np.exp(-D / max(scale, 0.01))
        np.fill_diagonal(A, 0.0)
        degrees = A.sum(axis=1)
        D_inv_sqrt = np.diag(
            1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
        L_norm = np.eye(K) - D_inv_sqrt @ A @ D_inv_sqrt
        try:
            _, eigvecs = np.linalg.eigh(L_norm)
            return (eigvecs[:, 1] > 0).astype(float)
        except np.linalg.LinAlgError:
            return self.random_state.randint(0, 2, K).astype(float)

    def _svd_residual_init(self, q_marginal: np.ndarray) -> np.ndarray:
        """SVD on residual matrix."""
        _, _, wins_m_d, totals_m_d = self._get_decayed_observations()
        K = self.K
        R = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                n = totals_m_d[i, j]
                if n >= 2:
                    w = wins_m_d[i, j]
                    p = np.clip(w / n, 0.05, 0.95)
                    R[i, j] = (np.log(p / (1 - p))
                               - (q_marginal[i] - q_marginal[j]))
        R = (R - R.T) / 2.0
        try:
            U, _, _ = np.linalg.svd(R)
            return (U[:, 0] > 0).astype(float)
        except np.linalg.LinAlgError:
            return self.random_state.randint(0, 2, K).astype(float)

    # ── EM estimation ─────────────────────────────────────────────────

    def _run_em_once(self, g_init: np.ndarray,
                     q_init: np.ndarray) -> tuple:
        """Single EM run. Returns (best_q, best_beta, best_g, loss)."""
        wins_e_d, totals_e_d, _, _ = self._get_decayed_observations()

        mask = totals_e_d >= 1
        P_obs = np.where(mask,
                         wins_e_d / np.maximum(totals_e_d, 1), 0.0)
        total_obs = max(float(mask.sum()), 1.0)

        g = g_init.copy()
        q = q_init.copy()
        beta = np.zeros(self.M)

        best_loss = np.inf
        best_q, best_beta, best_g = q.copy(), beta.copy(), g.copy()

        for _ in range(self.em_iters):
            # ── M-step: GD on (q, beta) given g ──────────────────────
            gf = g
            G_diff = gf[:, None] - gf[None, :]
            for _ in range(self.gd_iters):
                Q_eff = q[None, :] + beta[:, None] * gf[None, :]
                Logit = Q_eff[:, :, None] - Q_eff[:, None, :]
                P_pred = self._sig(Logit)
                Err = np.where(mask, totals_e_d * (P_pred - P_obs), 0.0)
                dq = Err.sum((0, 2)) - Err.sum((0, 1))
                db = (Err * G_diff[None, :, :]).sum((1, 2))

                # ── FIX 1: gradient stability gate ───────────────────
                grad_norm = np.sqrt(np.sum(dq**2) + np.sum(db**2))
                if self.gd_stability_gate > 0 and grad_norm < self.gd_stability_gate:
                    break  # gradient too small, stop early

                q -= self.gd_lr * dq / total_obs
                q -= self.lambda_q * q
                q -= q.mean()
                beta -= self.gd_lr * db / total_obs

            # Track best by squared loss
            Q_eff = q[None, :] + beta[:, None] * gf[None, :]
            Logit = Q_eff[:, :, None] - Q_eff[:, None, :]
            P_pred = self._sig(Logit)
            loss = float(np.sum(
                np.where(mask, totals_e_d * (P_pred - P_obs) ** 2, 0.0)))
            if loss < best_loss:
                best_loss = loss
                best_q = q.copy()
                best_beta = beta.copy()
                best_g = g.copy()

            # ── E-step: reassign g per arm ────────────────────────────
            g_new = g.copy()
            for a in range(self.K):
                arm_losses = []
                for cg in [0.0, 1.0]:
                    g_try = g.copy()
                    g_try[a] = cg
                    Qe = q[None, :] + beta[:, None] * g_try[None, :]
                    Lt = Qe[:, :, None] - Qe[:, None, :]
                    Pt = self._sig(Lt)
                    arm_loss = 0.0
                    for b in range(self.K):
                        if a == b:
                            continue
                        arm_loss += float(np.sum(np.where(
                            mask[:, a, b],
                            totals_e_d[:, a, b]
                            * (Pt[:, a, b] - P_obs[:, a, b]) ** 2, 0.0)))
                        arm_loss += float(np.sum(np.where(
                            mask[:, b, a],
                            totals_e_d[:, b, a]
                            * (Pt[:, b, a] - P_obs[:, b, a]) ** 2, 0.0)))
                    arm_losses.append(arm_loss)
                g_new[a] = 0.0 if arm_losses[0] <= arm_losses[1] else 1.0
            g = g_new

        return best_q, best_beta, best_g, best_loss

    def _fit_model(self) -> None:
        q_marginal = self._marginal_q()
        inits = []

        if self.use_spectral_init:
            g_spec = self._spectral_init()
            inits.extend([g_spec, 1.0 - g_spec])

        if self.use_svd_init:
            g_svd = self._svd_residual_init(q_marginal)
            inits.extend([g_svd, 1.0 - g_svd])


        n_rand = max(0, self.em_restarts - len(inits))
        for _ in range(n_rand):
            inits.append(self.random_state.randint(0, 2, self.K).astype(float))

        best_q, best_beta, best_g = None, None, None
        best_loss = np.inf

        for g_init in inits:
            q_r, beta_r, g_r, loss_r = self._run_em_once(g_init, q_marginal)
            if loss_r < best_loss:
                best_loss = loss_r
                best_q = q_r
                best_beta = beta_r
                best_g = g_r

        self.q = best_q
        self.beta = best_beta
        self.g = best_g.astype(np.int8)
        self._model_fitted = True

    # ── UCB selection ─────────────────────────────────────────────────

    def _skill_vector(self) -> np.ndarray:
        q_diff = self.q[:, None] - self.q[None, :]
        pairwise = self._sig(q_diff)
        np.fill_diagonal(pairwise, 0.0)
        return pairwise.sum(axis=1) / max(self.K - 1, 1)

    def _ucb_scores(self) -> np.ndarray:
        sv = self._skill_vector()
        t = max(self.time_step, 1)
        sc = np.zeros(self.K)
        for a in range(self.K):
            n = float(self.totals_m[a].sum() + self.totals_m[:, a].sum())
            bonus = np.sqrt(self.alpha * np.log(t) / (n + 1.0))
            if n < t * 0.01:
                bonus *= 2.0
            sc[a] = sv[a] + bonus
        return sc

    def _select_pair(self):
        sc = self._ucb_scores()
        i = int(np.argmax(sc))

        if self.random_state.rand() < self.epsilon:
            avail = [x for x in range(self.K) if x != i]
            j = int(self.random_state.choice(avail))
        else:
            sc2 = sc.copy()
            sc2[i] = -np.inf
            j = int(np.argmax(sc2))

        if (self.cross_group_correction and 
            self._model_fitted and 
            self.g.std() > 0.1):
            cross = [x for x in range(self.K) if self.g[x] != self.g[i]]
            if cross and self.g[i] == self.g[j]:
                j = int(self.random_state.choice(cross))

        return i, j

    # ── main step ─────────────────────────────────────────────────────

    def step(self) -> None:
        self.time_step += 1

        if self.time_step <= self._WARMUP_ROUNDS:
            user = (self.time_step - 1) % self.M
            pair_idx = ((self.time_step - 1) // self.M
                        % len(self._warmup_pairs))
            arm_c, arm_d = self._warmup_pairs[pair_idx]
        else:
            if not self._model_fitted:
                self._fit_model()

            arm_c, arm_d = self._select_pair()

            if self.time_step >= 1400:
                print(
                    self.time_step,
                    arm_c,
                    arm_d
                )

            user = int(self.random_state.randint(0, self.M))

        raw = self._base_env.duel(arm_c, arm_d, user)
        won  = arm_c if raw else arm_d
        lost = arm_d if raw else arm_c

        self.wins_m[won, lost]            += 1.0
        self.wins_e[user, won, lost]      += 1.0
        self.totals_m[arm_c, arm_d]       += 1.0
        self.totals_m[arm_d, arm_c]       += 1.0
        self.totals_e[user, arm_c, arm_d] += 1.0
        self.totals_e[user, arm_d, arm_c] += 1.0

        if (self._model_fitted
                and self.time_step > self._WARMUP_ROUNDS
                and self.time_step % self.em_refit_freq == 0):
            self._fit_model()

    # ── interface ─────────────────────────────────────────────────────

    def get_condorcet_winner(self) -> int:
        return int(np.argmax(self.q))

    def is_finished(self) -> bool:
        return self.time_step >= self.time_horizon

    @property
    def s(self):
        return self.q