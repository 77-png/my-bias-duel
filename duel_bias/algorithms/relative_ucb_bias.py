"""An implementation of the Relative Upper Confidence Bound (RCB) algorithm."""

from typing import Optional
import numpy as np

from duelpy.algorithms.relative_ucb import RelativeUCB
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duelpy.feedback import FeedbackMechanism
from duelpy.util.utility_functions import argmax_set
from duel_bias.model import UserBiasDueling
from duel_bias.feedback import FastPreferenceEstimateBias
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism


class RelativeUCBBias(BiasAlgorithm, RelativeUCB):

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        exploratory_constant: float = 0.51,
        random_state: Optional[np.random.RandomState] = None,
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

        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)

        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        self.hypothesized_arm: Optional[int] = None
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        obj = feedback_mechanism

        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism

        if hasattr(obj, 'user_bias'):
            self.num_users = len(obj.user_bias)
        else:
            self.num_users = 1

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

        upper_confidence_bound_matrix = self.preference_estimate.get_upper_estimate_matrix()
        potential_arms = set()
        for arm_i in range(self.wrapped_feedback.get_num_arms()):
            could_be_winner = True
            for arm_j in range(self.wrapped_feedback.get_num_arms()):
                if upper_confidence_bound_matrix[arm_i][arm_j] < 0.5:
                    could_be_winner = False
            if could_be_winner:
                potential_arms.add(arm_i)

        if len(potential_arms) == 0:
            arm_c = self.random_state.choice(
                range(self.wrapped_feedback.get_num_arms())
            )
            potential_arms.add(arm_c)

        if (
            self.hypothesized_arm is not None
            and self.hypothesized_arm not in potential_arms
        ):
            self.hypothesized_arm = None

        arm_c = self._select_potential_winner(potential_arms)
        upper_confidences = upper_confidence_bound_matrix[:, arm_c]
        arm_d = self.random_state.choice(argmax_set(upper_confidences, [arm_c]))
        user = self.choose_feedback_user()

        self.preference_estimate.enter_sample(
            arm_c, arm_d, self.wrapped_feedback.duel(arm_c, arm_d, user), user
        )
    
    def get_user_bias_estimates(self):
        return self.preference_estimate.get_user_bias()

    def get_arm_score_estimates(self):
        return self.preference_estimate.get_arm_scores()

    def get_preference_probability(self, arm_i: int, arm_j: int, user: Optional[int] = None):
        return self.preference_estimate.get_preference_probability(arm_i, arm_j, user)

    def get_mean_estimate_matrix(self):
        return self.preference_estimate.get_mean_estimate_matrix()