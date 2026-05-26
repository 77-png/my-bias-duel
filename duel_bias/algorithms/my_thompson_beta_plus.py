from typing import Optional
import numpy as np
from scipy.optimize import minimize_scalar
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duelpy.feedback import FeedbackMechanism
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism
from duel_bias.model import UserBiasDueling
from duel_bias.feedback import PreferenceEstimateBias

PRIOR = 0.5

class MyThompsonBetaPlus(BiasAlgorithm, DoubleThompsonSampling):
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.default_rng = None,
        c_base: float = 0.51,
        assigned_user_bias: Optional[np.ndarray] = None,
        comparative_user_bias: Optional[np.ndarray] = None,
        is_track_residual: Optional[bool] = False,
    ):
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)

        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)
        obj = feedback_mechanism
        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism
        self.num_users = len(obj.user_bias)
        self.random_state = random_state if random_state is not None else np.random.default_rng(42)
        self.c_base = c_base
        self.time_step = 0

        K = self.wrapped_feedback.get_num_arms()
        self.wins = np.zeros((K, K, self.num_users), dtype=int)
        self.trials = np.zeros_like(self.wins, dtype=int)
        self.r_hat = np.ones(self.num_users)
        self.n_user = np.zeros(self.num_users, dtype=int)

        self.preference_estimate = PreferenceEstimateBias(
            num_arms=K,
            num_users=self.num_users,
            assigned_user_bias=assigned_user_bias,
        )
        self.choose_feedback_user = UserBiasDueling(self.num_users).choose_feedback_user

    def _update_user_bias(self, user: int, i: int, j: int, outcome: bool) -> None:
        w = self.wins[i, j, user]
        n = self.trials[i, j, user]
        if n == 0:
            return
        pm = w / n
        ri = pm * (1 - 0.5) / (0.5 * (1 - pm) + 1e-8)
        self.r_hat[user] = np.clip(ri, 0.1, 2.0)
        self.n_user[user] += 1

    def step(self) -> None:
        self.time_step += 1
        self._update_confidence_radius()

        user = self.choose_feedback_user()
        adaptive_c = self.c_base + 0.3 * (1 - self.r_hat[user])
        i = self._sample_arm(user, adaptive_c)
        j = self._sample_second_arm(i, user, adaptive_c)

        outcome = self.wrapped_feedback.duel(i, j, user)
        self.wins[i, j, user] += outcome
        self.wins[j, i, user] += 1 - outcome
        self.trials[i, j, user] += 1
        self.trials[j, i, user] += 1
        self.preference_estimate.enter_sample(i, j, outcome, user)

        self._update_user_bias(user, i, j, outcome)

    def _sample_arm(self, user: int, c: float) -> int:
        K = self.wrapped_feedback.get_num_arms()
        sample = np.zeros(K)
        dynamic_prior = PRIOR / (self.n_user[user] + 1)
        min_shape = 1e-2 

        for i in range(K):
            a = c + dynamic_prior
            b = c + dynamic_prior
            for j in range(K):
                if j == i:
                    continue
                w = self.wins[i, j, user]
                n = self.trials[i, j, user]
                a += w
                b += n - w
            a = max(a, min_shape)
            b = max(b, min_shape)
            sample[i] = self.random_state.beta(a, b)
        return int(np.argmax(sample))

    def _sample_second_arm(self, first: int, user: int, c: float) -> int:
        K = self.wrapped_feedback.get_num_arms()
        sample = np.zeros(K)
        dynamic_prior = PRIOR / (self.n_user[user] + 1)
        min_shape = 1e-2

        for j in range(K):
            if j == first:
                continue
            w = self.wins[first, j, user]
            n = self.trials[first, j, user]
            a = w + c + dynamic_prior
            b = n - w + c + dynamic_prior
            a = max(a, min_shape)
            b = max(b, min_shape)
            sample[j] = self.random_state.beta(a, b)
        return int(np.argmax(sample))

    def get_condorcet_winner(self) -> int:
        p_mat = self.preference_estimate.get_preference_estimate()
        return int(np.argmax(np.mean(p_mat, axis=1)))

    def is_finished(self) -> bool:
        return self.time_step >= self.time_horizon