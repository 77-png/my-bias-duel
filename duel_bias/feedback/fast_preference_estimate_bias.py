"""Fast implementation of BARP model for bias-aware preference estimation"""
import numpy as np
from typing import Optional

from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate
from duelpy.stats.preference_matrix import PreferenceMatrix


class FastPreferenceEstimateBias(PreferenceEstimate):
    """Fast BARP implementation using mini-batch alternating MLE (keeps original public API).

    Changes from original:
      - Use correct BARP likelihood gradient: grad = (pred - y)
      - Alternating updates: several steps for scores (s) keeping theta fixed,
        then several steps for theta keeping s fixed.
      - Vectorized mini-batch processing for stability.
      - Mean-estimate computed by averaging per-user pairwise probs (as BARP does).
      - Keeps external API: enter_sample, get_mean_estimate_matrix, sample_preference_matrix, ...
    """
    def __init__(
        self,
        num_arms: int,
        num_users: int,
        confidence_radius: ConfidenceRadius = TrivialConfidenceRadius(),
        assigned_user_bias: Optional[np.array] = None,
        group_indicator: Optional[np.array] = None,
        reference_group: int = 0,
        learning_rate: float = 0.001,
        reg_strength: float = 1e-4,
        batch_size: int = 64,
        start_update_theta_after: int = 100,
        # new hyperparams for alternating and Adam stability
        steps_s: int = 3,
        steps_theta: int = 3,
        lr_s: Optional[float] = None,
        lr_theta: Optional[float] = None,
        clamp_theta: float = 20.0,
    ):
        super().__init__(num_arms, confidence_radius)
        self.num_users = num_users

        # Group indicator (binary/group reference) - keep original API
        if group_indicator is None:
            self.group_indicator = np.zeros(num_arms, dtype=int)
        else:
            self.group_indicator = group_indicator
        self.reference_group = reference_group

        # MODEL PARAMETERS
        self.scores = np.zeros(num_arms)              # s_i
        # keep theta as per-user scalar (binary-group case), consistent with your original design
        self.theta = np.zeros(num_users)              # theta_k

        # optimization hyperparams
        # keep compatibility: if lr_s/lr_theta not provided, use learning_rate for both
        self.lr = learning_rate
        self.lr_s = learning_rate if lr_s is None else lr_s
        self.lr_theta = learning_rate if lr_theta is None else lr_theta
        self.reg_strength = reg_strength
        self.batch_size = int(batch_size)
        self.start_update_theta_after = int(start_update_theta_after)

        # alternating steps
        self.steps_s = int(steps_s)
        self.steps_theta = int(steps_theta)

        # precompute delta vector (binary indicator whether arm belongs to reference group)
        self.delta_vector = (self.group_indicator == reference_group).astype(float)

        # statistics
        self.wins = np.zeros((num_users, num_arms, num_arms), dtype=np.int32)
        self.totals = np.zeros((num_users, num_arms, num_arms), dtype=np.int32)
        self.total_comparisons = 0

        # caches
        self._cached_mean_estimate = np.full((num_arms, num_arms), 0.5)
        self._cached_radius = None

        # assigned bias for testing (if provided, force theta to this)
        if assigned_user_bias is not None:
            arr = np.array(assigned_user_bias)
            if arr.ndim == 1:
                assert arr.shape[0] == num_users
            else:
                raise ValueError("assigned_user_bias must be 1-D array of length num_users")
        self._assigned_user_bias = assigned_user_bias

        # Adam optimizer state for scores and theta
        self.adam_m_s = np.zeros_like(self.scores)
        self.adam_v_s = np.zeros_like(self.scores)
        self.adam_m_theta = np.zeros_like(self.theta)
        self.adam_v_theta = np.zeros_like(self.theta)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8

        # clamp for theta numeric stability
        self.clamp_theta = clamp_theta

        # sample buffer (mini-batch)
        self.sample_buffer = []
        self.buffer_limit = max(4 * self.batch_size, 1000)

    def enter_sample(
        self, first_arm_index: int, second_arm_index: int, first_won: bool, user_feedback: int = None
    ) -> None:
        if user_feedback is None:
            raise ValueError("user_feedback cannot be None")

        i, j, k = int(first_arm_index), int(second_arm_index), int(user_feedback)
        y = 1 if first_won else 0

        # update wins/totals (same as before)
        if first_won:
            self.wins[k, i, j] += 1
        else:
            self.wins[k, j, i] += 1
        self.totals[k, i, j] += 1
        self.totals[k, j, i] += 1

        self.total_comparisons += 1

        # add to buffer (store y as float 0/1)
        self.sample_buffer.append((i, j, k, float(y)))

        # if buffer large enough or periodic flushing, perform alternating mini-batch update
        if len(self.sample_buffer) >= self.batch_size or (self.total_comparisons % max(100, self.batch_size) == 0):
            # protect against huge buffer growth
            if len(self.sample_buffer) > self.buffer_limit:
                # keep only the most recent buffer_limit samples
                self.sample_buffer = self.sample_buffer[-self.buffer_limit :]
            self._fast_gradient_update_alternating()
            # clear buffer after update
            self.sample_buffer.clear()
            # invalidate mean/radius cache so next queries reflect updated params
            self._cached_radius = None
            self._update_mean_estimate_fast()

        # occasionally refresh mean estimate/radius for long idle periods
        if self.total_comparisons % 500 == 0:
            self._update_mean_estimate_fast()
            self._cached_radius = None

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid (vectorized)
        x_clip = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def _fast_gradient_update_alternating(self):
        """Mini-batch alternating MLE update (correct BARP gradients)."""
        if not self.sample_buffer:
            return

        # If fixed assigned bias, just set theta and skip theta updates
        if self._assigned_user_bias is not None:
            self.theta = np.array(self._assigned_user_bias, copy=True)

        # convert batch to numpy arrays
        bi = np.array([t[0] for t in self.sample_buffer], dtype=int)
        bj = np.array([t[1] for t in self.sample_buffer], dtype=int)
        bk = np.array([t[2] for t in self.sample_buffer], dtype=int)
        by = np.array([t[3] for t in self.sample_buffer], dtype=float)
        B = len(bi)

        # precompute group indicators for arms in batch
        delta_i = self.delta_vector[bi]    # (B,)
        delta_j = self.delta_vector[bj]    # (B,)
        delta_diff = delta_i - delta_j     # (B,)

        # Helper to compute per-sample score diffs s_i + theta_k * delta_i - (s_j + theta_k * delta_j)
        def compute_diff(scores, theta_vec):
            # theta_vec indexed by bk: per-sample theta
            t_k = theta_vec[bk]               # (B,)
            si = scores[bi] + t_k * delta_i
            sj = scores[bj] + t_k * delta_j
            return si - sj

        # ---------- update scores: several steps with theta fixed ----------
        for step in range(self.steps_s):
            diffs = compute_diff(self.scores, self.theta)    # (B,)
            preds = self._sigmoid(diffs)                     # (B,)
            err = preds - by                                 # (B,)  (pred - y), correct MLE gradient

            # accumulate gradients for scores: grad_s[i] += err, grad_s[j] -= err
            grad_s = np.zeros_like(self.scores)
            # vectorized accumulation using np.add.at
            np.add.at(grad_s, bi, err)
            np.add.at(grad_s, bj, -err)

            # regularization
            grad_s += self.reg_strength * self.scores

            # Adam update for scores
            t = max(1, self.total_comparisons)
            self.adam_m_s = self.adam_beta1 * self.adam_m_s + (1 - self.adam_beta1) * grad_s
            self.adam_v_s = self.adam_beta2 * self.adam_v_s + (1 - self.adam_beta2) * (grad_s ** 2)
            m_hat_s = self.adam_m_s / (1 - self.adam_beta1 ** t)
            v_hat_s = self.adam_v_s / (1 - self.adam_beta2 ** t)
            step_s = self.lr_s * m_hat_s / (np.sqrt(v_hat_s) + self.adam_eps)
            self.scores -= step_s

        # ---------- update theta (per-user scalar) if allowed ----------
        if self.total_comparisons >= self.start_update_theta_after and self._assigned_user_bias is None:
            # We'll accumulate gradient per user over the batch
            for step in range(self.steps_theta):
                diffs = compute_diff(self.scores, self.theta)
                preds = self._sigmoid(diffs)
                err = preds - by   # (B,)

                # grad_theta[k] = sum_{samples by user k} err[sample] * (delta_i - delta_j)
                grad_theta = np.zeros_like(self.theta)  # (num_users,)
                # accumulate using python loop over unique users in batch for efficiency when users per batch small
                unique_users, inverse_idx = np.unique(bk, return_inverse=True)
                # sum_err_per_user = np.bincount(inverse_idx, weights=err)
                # but we need weighted by delta_diff per sample (per-sample weight)
                for idx_u, u in enumerate(unique_users):
                    mask = (bk == u)
                    # sum of err * delta_diff for this user
                    agg = np.dot(err[mask], delta_diff[mask])
                    grad_theta[u] = agg

                # regularization
                grad_theta += self.reg_strength * self.theta

                # Adam update for theta (vector)
                t = max(1, self.total_comparisons)
                self.adam_m_theta = self.adam_beta1 * self.adam_m_theta + (1 - self.adam_beta1) * grad_theta
                self.adam_v_theta = self.adam_beta2 * self.adam_v_theta + (1 - self.adam_beta2) * (grad_theta ** 2)
                m_hat_t = self.adam_m_theta / (1 - self.adam_beta1 ** t)
                v_hat_t = self.adam_v_theta / (1 - self.adam_beta2 ** t)
                step_t = self.lr_theta * m_hat_t / (np.sqrt(v_hat_t) + self.adam_eps)
                self.theta -= step_t

                # clamp theta for numeric stability
                self.theta = np.clip(self.theta, -self.clamp_theta, self.clamp_theta)
        else:
            # if theta updates are not active or assigned bias provided, ensure assigned bias is kept
            if self._assigned_user_bias is not None:
                self.theta = np.array(self._assigned_user_bias, copy=True)

        # after updates, invalidate caches
        self._cached_mean_estimate = np.full((self.num_arms, self.num_arms), 0.5)
        self._cached_radius = None

    def _update_mean_estimate_fast(self):
        """Compute mean estimate by averaging per-user pairwise probabilities (BARP marginalization)."""
        n = self.num_arms
        if n == 0:
            return
        # Build per-user effective scores: for binary-group case, score_user[i,k] = s_i + theta_k * delta_i
        # score_user shape: (n, num_users)
        score_user = np.expand_dims(self.scores, axis=1) + np.outer(self.delta_vector, self.theta)  # (n, num_users)
        # compute per-user pairwise probabilities P_k(i>j) and average across users
        # Efficient vectorization: for each user, compute outer diff matrix then sigmoid and average
        n_users = self.num_users
        # To avoid huge memory when n large, compute pairwise row-by-row
        mean_mat = np.empty((n, n), dtype=float)
        for i in range(n):
            # diff to all j for all users: shape (n, n_users)
            diffs = score_user[i, None, :] - score_user[:, :]  # (n, n_users)
            # apply sigmoid elementwise and average over users
            probs = 1.0 / (1.0 + np.exp(-np.clip(diffs, -50.0, 50.0)))
            # probs.shape = (n, n_users) where probs[j, k] = P_k(i > j)
            mean_mat[i, :] = np.mean(probs, axis=1)
        np.fill_diagonal(mean_mat, 0.5)
        self._cached_mean_estimate = np.clip(mean_mat, 1e-3, 1 - 1e-3)

    def get_mean_estimate_matrix(self):
        return self._cached_mean_estimate.copy()

    def get_radius_matrix(self):
        """Confidence radius aggregated across users (improved simple version)."""
        if self._cached_radius is None:
            # total comparisons between i and j across all users
            num_samples = np.sum(self.totals + np.transpose(self.totals, axes=(0, 2, 1)), axis=0)
            base = 1.0 / np.sqrt(num_samples + 1.0)
            np.fill_diagonal(base, 0.0)
            self._cached_radius = np.clip(base, 0.0, 0.5)
        return self._cached_radius

    def sample_preference_matrix(self, random_state: np.random.RandomState) -> PreferenceMatrix:
        """Sample preference matrix using Beta sampling aggregated across users (like original)."""
        n = self.num_arms
        if n == 0:
            return PreferenceMatrix(np.zeros((0, 0)))
        beta_a = np.sum(self.wins, axis=0) + 1.0
        beta_b = np.sum(self.totals, axis=0) - np.sum(self.wins, axis=0) + 1.0
        beta_a = np.clip(beta_a, 1e-6, 1e6)
        beta_b = np.clip(beta_b, 1e-6, 1e6)
        upper = random_state.beta(beta_a, beta_b)
        # symmetrize
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                p = upper[i, j]
                q = upper[j, i]
                avg = 0.5 * (p + (1.0 - q))
                avg = np.clip(avg, 1e-3, 1 - 1e-3)
                mat[i, j] = avg
                mat[j, i] = 1.0 - avg
        np.fill_diagonal(mat, 0.5)
        return PreferenceMatrix.from_upper_triangle(mat)

    def get_preference_probability(self, i: int, j: int, user: Optional[int] = None) -> float:
        """Return preference probability using current estimates (numerically stable)."""
        if i == j:
            return 0.5
        if user is None:
            score_diff = self.scores[i] - self.scores[j]
        else:
            delta_i = self.delta_vector[i]
            delta_j = self.delta_vector[j]
            score_i = self.scores[i] + self.theta[user] * delta_i
            score_j = self.scores[j] + self.theta[user] * delta_j
            score_diff = score_i - score_j
        if score_diff > 50:
            return 1.0
        if score_diff < -50:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-score_diff)))

    def get_user_bias(self) -> np.array:
        return self.theta.copy()

    def get_arm_scores(self) -> np.array:
        return self.scores.copy()

    def get_num_samples(self, i: int, j: int, user_index: int = None) -> int:
        if user_index is None:
            return int(np.sum(self.totals[:, i, j]))
        else:
            return int(self.totals[user_index, i, j])

    def get_estimated_probabilities(self) -> np.ndarray:
        """
        返回所有arm对的估计胜率矩阵（K×K）
        返回值：
            p_ij: [i,j] 表示arm i击败arm j的估计概率
        """
        # 如果有预计算的概率矩阵，直接返回
        if hasattr(self, 'p_ij_estimates') and self.p_ij_estimates is not None:
            return self.p_ij_estimates.copy()
        
        # 如果没有预计算，基于当前模型参数计算（通用实现）
        K = self.num_arms
        p_ij = np.zeros((K, K), dtype=np.float32)
        
        # 遍历所有arm对
        for i in range(K):
            for j in range(K):
                if i == j:
                    p_ij[i, j] = 0.5
                    continue
                
                # 基于分组和theta计算估计胜率（和MyGroup逻辑一致）
                gi = self.group_indicator[i] if hasattr(self, 'group_indicator') else 0
                gj = self.group_indicator[j] if hasattr(self, 'group_indicator') else 0
                
                # 取所有用户的平均theta（或用默认theta）
                theta_avg = np.mean(self.theta) if hasattr(self, 'theta') and len(self.theta) > 0 else 0.0
                
                exp_si = np.exp(np.clip(self.s[i] + theta_avg * gi, -10, 10)) if hasattr(self, 's') else np.exp(0)
                exp_sj = np.exp(np.clip(self.s[j] + theta_avg * gj, -10, 10)) if hasattr(self, 's') else np.exp(0)
                
                p_ij[i, j] = exp_si / (exp_si + exp_sj + 1e-8)
        
        # 缓存结果（可选）
        self.p_ij_estimates = p_ij
        return p_ij.copy()
