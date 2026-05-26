from typing import Optional
import numpy as np

from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.feedback import FeedbackMechanism
from duel_bias.model import UserBiasDueling
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism

from duel_bias.feedback import FastPreferenceEstimateBias


class DoubleThompsonSamplingBias(BiasAlgorithm, DoubleThompsonSampling):
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.default_rng = None,
        exploratory_constant: float = 0.51,
        assigned_user_bias: Optional[np.array] = None,
        comparative_user_bias: Optional[np.array] = None,
        is_track_residual: Optional[bool] = False,
        group_indicator: Optional[np.array] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ):
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)

        # Reassign object of feedback to receive a feedback with user identifier
        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)

        # Get number of users
        obj = feedback_mechanism

        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism

        if hasattr(obj, 'user_bias'):
            self.num_users = len(obj.user_bias)
        else:
            raise ValueError('An unexpected issue has arisen: user bias is not found.')

        self.time_step = 0
        self.exploratory_constant = exploratory_constant
        self.random_state = (
            random_state if random_state is not None else np.random.default_rng(42)
        )

        num_arms = self.wrapped_feedback.get_num_arms()
        if group_indicator is None:
            group_indicator = np.zeros(num_arms)
        
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
        """Run one round of an algorithm."""
        self.time_step += 1
        self._update_confidence_radius()
        arm_c = self._choose_first_candidate()
        arm_d = self._choose_second_candidate(arm_c)
        user = self.choose_feedback_user()
        
        result = self.wrapped_feedback.duel(arm_c, arm_d, user)
        
        self.preference_estimate.enter_sample(arm_c, arm_d, result, user)