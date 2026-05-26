from typing import Optional
import numpy as np
from collections import defaultdict
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.feedback import FeedbackMechanism
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism
from duel_bias.model import UserBiasDueling
from duelpy.algorithms.interfaces import CondorcetProducer
from duel_bias.feedback import PreferenceEstimateBias


class MyBTBiasAlgo(BiasAlgorithm, CondorcetProducer):
    def _extract_num_users(self) -> int:
        obj = self.wrapped_feedback
        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism
        if hasattr(obj, 'user_bias'):
            return len(obj.user_bias)
        raise ValueError('User bias not found in the wrapped feedback chain.')

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.default_rng = None,
        assigned_user_bias: Optional[np.ndarray] = None,
        comparative_user_bias: Optional[np.ndarray] = None,
        is_track_residual: Optional[bool] = False,
        lr: float = 0.01,
        num_step: int = 10,
        start_update_theta_after: int = 20,
    ):
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)
        
        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)
        self.random_state = random_state if random_state is not None else np.random.default_rng(42)
        self.lr = lr
        self.num_step = num_step
        self.start_update_theta_after = start_update_theta_after
        self.time_step = 0

        num_arms = self.wrapped_feedback.get_num_arms()
        num_users = self._extract_num_users()

        self.K = num_arms
        self.M = num_users

        # 初始化参数（与 full_optimized.py 一致）
        self.s = np.random.randn(self.K)          # item skills
        self.theta = np.random.randn(self.M)      # user biases
        self.g = np.random.randint(0, 2, size=self.K)  # item groups (0 or 1)

        self.counts = defaultdict(int)   # (i, j, user) -> wins for i
        self.totals = defaultdict(int)   # (i, j, user) -> total duels

        self.choose_feedback_user = UserBiasDueling(self.M).choose_feedback_user

        self.preference_estimate = PreferenceEstimateBias(
            num_arms=num_arms,
            num_users=num_users,
            assigned_user_bias=assigned_user_bias,
        )

    def _true_p_ij(self, si, sj):
        """无偏 Bradley-Terry probability"""
        return np.exp(si) / (np.exp(si) + np.exp(sj))

    def _biased_p_ij(self, i, j, k):
        """带用户偏置的成对偏好概率"""
        si, sj = self.s[i], self.s[j]
        gi, gj = self.g[i], self.g[j]
        th = self.theta[k]
        exp_i = np.exp(si + th * gi)
        exp_j = np.exp(sj + th * gj)
        return exp_i / (exp_i + exp_j)

    def _update_group_assignments(self):
        """与 full_optimized.py 完全一致的分组更新逻辑"""
        new_g = self.g.copy()
        for i in range(self.K):
            losses = []
            for trial_gi in [0, 1]:
                g_trial = self.g.copy()
                g_trial[i] = trial_gi
                loss = 0.0
                for k in range(self.M):
                    for j in range(self.K):
                        if i == j:
                            continue
                        n = self.totals[(i, j, k)]
                        if n == 0:
                            continue
                        observed = self.counts[(i, j, k)] / n

                        gi, gj = g_trial[i], g_trial[j]
                        gamma_i = np.exp(self.theta[k] * gi)
                        gamma_j = np.exp(self.theta[k] * gj)
                        pk = gamma_i * np.exp(self.s[i]) / (
                            gamma_i * np.exp(self.s[i]) + gamma_j * np.exp(self.s[j])
                        )
                        loss += (pk - observed) ** 2
                losses.append(loss)
            new_g[i] = 0 if losses[0] < losses[1] else 1
        return new_g

    def _update_parameters(self):
        """与 full_optimized.py 一致的梯度更新（无 momentum）"""
        update_theta = self.time_step >= self.start_update_theta_after

        for _ in range(self.num_step):
            grad_s = np.zeros(self.K)
            grad_theta = np.zeros(self.M) if update_theta else None

            for k in range(self.M):
                for i in range(self.K):
                    for j in range(self.K):
                        if i == j:
                            continue
                        n = self.totals[(i, j, k)]
                        if n == 0:
                            continue
                        freq = self.counts[(i, j, k)] / n
                        pred = self._biased_p_ij(i, j, k)
                        err = pred - freq
                        common = err * pred * (1 - pred)

                        grad_s[i] += common
                        grad_s[j] -= common

                        if update_theta:
                            grad_theta[k] += common * (self.g[i] - self.g[j])

            # 更新 s
            self.s -= self.lr * grad_s

            # 更新 theta
            if update_theta:
                self.theta -= self.lr * grad_theta

    def step(self) -> None:
        self.time_step += 1
        num_arms = self.wrapped_feedback.get_num_arms()

        # 简单 epsilon-greedy 探索（可替换为 UCB 等）
        epsilon = 0.1
        if self.random_state.random() < epsilon:
            i, j = self.random_state.choice(num_arms, 2, replace=False)
        else:
            # 基于当前 s 选择 top-2 arms
            skill = self.s.copy()
            i = int(np.argmax(skill))
            skill[i] = -np.inf
            j = int(np.argmax(skill))

        user = self.choose_feedback_user()
        outcome = self.wrapped_feedback.duel(i, j, user)

        # 更新计数
        if outcome == 1:
            self.counts[(i, j, user)] += 1
        else:
            self.counts[(j, i, user)] += 1
        self.totals[(i, j, user)] += 1
        self.totals[(j, i, user)] += 1

        # 执行参数更新（与 full_optimized.py 一致）
        self._update_parameters()
        self.g = self._update_group_assignments()

        # 更新 preference estimate for tracking
        self.preference_estimate.enter_sample(i, j, outcome, user)

    def get_condorcet_winner(self) -> int:
        # 返回 skill 最高的 arm（无偏估计）
        return int(np.argmax(self.s))

    def is_finished(self) -> bool:
        return self.time_step >= self.time_horizon