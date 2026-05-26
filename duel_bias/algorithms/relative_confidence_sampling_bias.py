"""An implementation of the Relative Confidence Sampling algorithm."""
from typing import Optional
import numpy as np

from duelpy.algorithms.relative_confidence_sampling import RelativeConfidenceSampling
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duel_bias.model import UserBiasDueling
from duel_bias.feedback import FastPreferenceEstimateBias
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism


class RelativeConfidenceSamplingBias(BiasAlgorithm, RelativeConfidenceSampling):
    """Bias-aware extension for the Relative Confidence Sampling algorithm.
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 0.501,
        assigned_user_bias: Optional[np.array] = None,
        comparative_user_bias: Optional[np.array] = None,
        is_track_residual: Optional[bool] = False,
        group_indicator: Optional[np.array] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)

        # Reassign object of feedback to receive a feedback with user identifier
        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)

        self.random_state = (
            random_state if random_state is not None else np.random.default_rng(42)
        )
        if exploratory_constant <= 0.5:
            raise ValueError("Value of exploratory constant must be greater than 0.5")
        self.exploratory_constant = exploratory_constant
        self.time_step = 0

        # Number of times each arm was already chosen as the champion
        self._champion_chosen_frequency = np.zeros(
            self.wrapped_feedback.get_num_arms(), dtype=int
        )

        # Get number of users
        obj = feedback_mechanism

        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism

        if hasattr(obj, 'user_bias'):
            self.num_users = len(obj.user_bias)
        else:
            # 默认值，如果找不到用户bias
            self.num_users = 1

        # 如果没有提供组别信息，创建默认值
        num_arms = self.wrapped_feedback.get_num_arms()
        if group_indicator is None:
            group_indicator = np.zeros(num_arms)
        
        # 使用FastPreferenceEstimateBias
        self.preference_estimate = FastPreferenceEstimateBias(
            num_arms=num_arms,
            num_users=self.num_users,
            assigned_user_bias=assigned_user_bias,
            group_indicator=group_indicator,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        self.choose_feedback_user = UserBiasDueling(self.num_users).choose_feedback_user

    def step(self) -> None:
        """Run one round of the algorithm."""
        self.time_step += 1
        # Update and set the new confidence radius in `preference_estimate` as per
        # the current `time_step`
        self._update_confidence_radius()

        champion = self._run_simulated_tournament()
        challenger = self._select_challenger_for(champion)
        user = self.choose_feedback_user()

        champion_won = self.wrapped_feedback.duel(champion, challenger, user)
        self.preference_estimate.enter_sample(champion, challenger, champion_won, user)
    
    # 添加一些有用的方法
    def get_user_bias_estimates(self):
        """获取用户bias估计"""
        return self.preference_estimate.get_user_bias()

    def get_arm_score_estimates(self):
        """获取项目分数估计"""
        return self.preference_estimate.get_arm_scores()

    def get_preference_probability(self, arm_i: int, arm_j: int, user: Optional[int] = None):
        """获取偏好概率"""
        return self.preference_estimate.get_preference_probability(arm_i, arm_j, user)

    def get_mean_estimate_matrix(self):
        """获取均值估计矩阵"""
        return self.preference_estimate.get_mean_estimate_matrix()