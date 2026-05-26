from typing import Optional
import numpy as np

from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.feedback import FeedbackMechanism
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism
from duel_bias.model import UserBiasDueling
from duelpy.algorithms.interfaces import CondorcetProducer

def _update_r_hat(p_minus_m: np.ndarray, p_ij_hat: float, r_hat: np.ndarray) -> np.ndarray:
    new_r = r_hat.copy()
    pm = np.clip(p_minus_m[0], 1e-3, 1 - 1e-3)
    ri = p_ij_hat * (1 - pm) / (pm * (1 - p_ij_hat) + 1e-8)
    new_r[0] = np.clip(ri, 0.1, 2.0)
    new_r[1] = np.clip(1.0 / ri, 0.1, 2.0) if ri != 0 else 1.0
    return new_r

def _update_p_hat(p_minus_m: np.ndarray, r_hat: np.ndarray, lr: float) -> float:
    p = 0.5
    for _ in range(50):
        denom = (1 - r_hat[0]) * p + r_hat[0]
        pred = p / denom
        grad = 2 * (pred - p_minus_m[0]) * (denom + (r_hat[0] - 1) * p) / (denom ** 2 + 1e-8)
        p -= lr * grad
        p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(p)

class MyBTBiasAlgo(BiasAlgorithm, CondorcetProducer):
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.Generator = None,
        assigned_user_bias: Optional[np.ndarray] = None,
        comparative_user_bias: Optional[np.ndarray] = None,
        is_track_residual: Optional[bool] = False,
        lr: float = 0.1,
        est_steps: int = 10,
    ):
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)

        # 框架已包好 BudgetedFeedbackMechanism，直接拿来用
        self.wrapped_feedback = feedback_mechanism
        self.random_state = random_state if random_state is not None else np.random.default_rng(42)
        self.lr = lr
        self.est_steps = est_steps

        num_arms = self.wrapped_feedback.get_num_arms()
        self.num_users = self._extract_num_users()
        self.wins = np.zeros((self.num_users, num_arms, num_arms), dtype=int)
        self.trials = np.zeros_like(self.wins)
        self.r_hat = np.ones((self.num_users, num_arms))
        self.p_ij_hat = 0.5
        self.choose_feedback_user = UserBiasDueling(self.num_users).choose_feedback_user

    # ---------- 辅助：把 user_bias 从包裹链里捞出来 ----------
    def _extract_num_users(self) -> int:
        obj = self.wrapped_feedback
        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism
        if hasattr(obj, 'user_bias'):
            return len(obj.user_bias)
        raise ValueError('User bias not found in the wrapped feedback chain.')

    # ---------- 每轮选臂 + 接收反馈 ----------
    def step(self) -> None:
        self.time_step += 1
        num_arms = self.wrapped_feedback.get_num_arms()

        # ε-greedy 选臂（可换成任意策略）
        epsilon = 0.1
        if self.random_state.random() < epsilon:
            i, j = self.random_state.choice(num_arms, 2, replace=False)
        else:
            skill = np.log(self.p_ij_hat / (1 - self.p_ij_hat + 1e-8))
            i = int(np.argmax(skill))
            j = int(np.argmax(skill + 1e-3 * self.random_state.random(num_arms)))
            if i == j:
                j = (i + 1) % num_arms

        user = self.choose_feedback_user()
        outcome = self.wrapped_feedback.duel(i, j, user)  # 1 若 i 胜 j

        # 更新统计
        self.wins[user, i, j] += outcome
        self.wins[user, j, i] += 1 - outcome
        self.trials[user, i, j] += 1
        self.trials[user, j, i] += 1

        self._estimate(user, i, j)

    # ---------- 你的估计器 ----------
    def _estimate(self, user: int, i: int, j: int) -> None:
        wins_ij = self.wins[user, i, j]
        trials_ij = self.trials[user, i, j]
        if trials_ij == 0:
            return

        p_minus_m = np.array([wins_ij / trials_ij], dtype=float)
        r_old = self.r_hat[user, [i, j]].copy()
        if r_old.size != 2:
            return

        for _ in range(self.est_steps):
            r_old = _update_r_hat(p_minus_m, self.p_ij_hat, r_old)
            self.p_ij_hat = _update_p_hat(p_minus_m, r_old, self.lr)

        self.r_hat[user, i] = r_old[0]
        self.r_hat[user, j] = r_old[1]

    # ---------- 返回 Condorcet winner ----------
    def get_condorcet_winner(self) -> int:
        skill = np.log(self.p_ij_hat / (1 - self.p_ij_hat + 1e-8))
        return int(np.argmax(skill)) 
    
if __name__ == "__main__":
    from scripts_for_experiments.benchmark_utils.bradley_terry_luce_model import BradleyTerryLuceModelBias
    from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism
    from duel_bias.model import UserBiasDueling
    import numpy as np

    rng = np.random.default_rng(42)
    env = BradleyTerryLuceModelBias(
        num_arms=5,
        user_bias=rng.beta(2, 1, 5),
        skill_vector=np.abs(rng.normal(0, 1, 5)),
        random_state=rng
    )
    ufd = UserBiasDueling(num_users=5)
    ufd.user_bias = env.user_bias
    ufd.feedback_mechanism = env
    fb = BudgetedFeedbackMechanism(ufd, max_duels=999999)

    algo = MyBTBiasAlgo(fb, time_horizon=100)
    for _ in range(10):
        algo.step()
    print("Local test passed.")