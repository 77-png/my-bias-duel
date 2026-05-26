"""An implementation of the UCB algorithm."""

from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duel_bias.utils.util import argmax_set_1, argmax_set_2

class UCB(CondorcetProducer):
    """Implementation of the UCB algorithm.
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        exploratory_constant: float = 0.51,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.preference_estimate = PreferenceEstimate(
            num_arms=self.wrapped_feedback.get_num_arms(),
        )

    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using the latest failure probability.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def get_condorcet_winner(self) -> Optional[int]:
        """Determine a Condorcet winner.
        """
        return (
            self.preference_estimate.get_mean_estimate_matrix().get_condorcet_winner()
        )

    def step(self) -> None:
        """Run one round of an algorithm."""
        self.time_step += 1
        self._update_confidence_radius()

        upper_confidence_bound_matrix = self.preference_estimate.get_upper_estimate_matrix()
        upper_confidences = np.sum(upper_confidence_bound_matrix.preferences, axis=1)
        arm_c, arm_d = argmax_set_2(upper_confidences)

        self.preference_estimate.enter_sample(
            arm_c, arm_d, self.wrapped_feedback.duel(arm_c, arm_d)
        )