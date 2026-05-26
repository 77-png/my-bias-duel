"""An implementation of the Bias-Aware UCB algorithm."""

from typing import Optional
import numpy as np

from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duel_bias.model import UserBiasDueling
from duel_bias.feedback import PreferenceEstimateBias
from duel_bias.feedback.bias_feedback_decorators import BudgetedFeedbackMechanism
from duel_bias.algorithms.BiasAlgorithm import BiasAlgorithm
from duel_bias.utils.util import argmax_set_1, argmax_set_2
from duel_bias.feedback.group_estimate_bias import GroupEstimateBias

class BiasSensitiveUCB(BiasAlgorithm, CondorcetProducer):
    """Implementation of the bias-aware UCB algorithm.
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        exploratory_constant: float = 0.51,
        random_state: Optional[np.random.RandomState] = None,
        assigned_user_bias: Optional[np.array] = None,
        comparative_user_bias: Optional[np.array] = None,
        init_exploration: Optional[str] = "random",
        greedy_epsilon: Optional[float] = 0.03,
        is_exploit: Optional[bool] = False,
        is_track_residual: Optional[bool] = False,
    ) -> None:
        """Initialize the BiasSensitiveUCB algorithm.

        Parameters:
            feedback_mechanism:
                Mechanism for receiving feedback from users
            time_horizon:
                Maximum number of time steps
            exploratory_constant:
                Constant for exploration term
            random_state:
                Random number generator
            assigned_user_bias:
                Pre-assigned user bias values
            comparative_user_bias:
                Comparative bias between users
            init_exploration:
                Initial exploration strategy (None, 'random' (default) or 'round-robin')
            greedy_epsilon:
                Probability of greedy exploration
            is_exploit:
                Whether to enable exploitation phase
            is_track_residual:
                Is tracking residuals of the biased evaluator dueling optimization problem
        """
        kwargs_bias_tracker = {
            "comparative_user_bias": comparative_user_bias,
            "compute_residual": is_track_residual,
        }
        super().__init__(kwargs_bias_tracker, feedback_mechanism, time_horizon)

        # Reassign object of feedback to receive a feedback with user identifier
        self.wrapped_feedback = BudgetedFeedbackMechanism(feedback_mechanism, time_horizon)
        num_arms = self.wrapped_feedback.get_num_arms()

        self.exploratory_constant = exploratory_constant
        self.time_step = 0
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        # Get number of users
        obj = feedback_mechanism

        while hasattr(obj, 'feedback_mechanism'):
            obj = obj.feedback_mechanism

        if hasattr(obj, 'user_bias'):
            self.num_users = len(obj.user_bias)
        else:
            raise ValueError('An unexpected issue has arisen: user bias is not found.')

        # self.preference_estimate = PreferenceEstimateBias(
        #     num_arms=num_arms,
        #     num_users=self.num_users,
        #     assigned_user_bias=assigned_user_bias,
        #     init_times=60 * num_arms * np.log(num_arms)
        # )
        self.preference_estimate = GroupEstimateBias(
            num_arms=num_arms,
            num_users=self.num_users
        )

        self.choose_feedback_user = UserBiasDueling(self.num_users).choose_feedback_user

        def random_exploration_generator():
            """Generator for random exploration strategy."""
            while True:
                arm_c, arm_d = self.random_state.choice(num_arms, size=2, replace=False)
                yield int(arm_c), int(arm_d)

        def round_robin_exploration_generator():
            """Generator for round-robin exploration strategy."""
            pairs = [(i, j) for i in range(num_arms) for j in range(num_arms) if j != i]
            current_pair_idx = 0

            while True:
                current_pair = pairs[current_pair_idx]
                current_pair_idx = (current_pair_idx + 1) % len(pairs)
                yield current_pair

        # Initial exploration
        if init_exploration is None:
            self.exploration_generator = None
            self.init_exploration_times = 0
        elif init_exploration == "random":
            self.exploration_generator = random_exploration_generator()
            self.init_exploration_times = int(5 * num_arms * np.log(num_arms))
        elif init_exploration == "round-robin":
            self.exploration_generator = round_robin_exploration_generator()
            self.init_exploration_times = 3 * int(num_arms * (num_arms - 1) / 2)
        else:
            raise ValueError("Invalid init_exploration strategy. Choose 'random' or 'round-robin'.")

        # Greedy exploration
        if greedy_epsilon > 0.1 or greedy_epsilon < 0:
            raise ValueError("Invalid greedy_epsilon value. Choose from [0, 0.1]")
        self.greedy_ep = greedy_epsilon
        
        # Decision stability tracking
        self.is_finish_explore = False
        self.is_exploit = is_exploit
        self.best_arm = None
        if self.is_exploit:
            self.stability_window_size = int(50 * num_arms * (np.log(num_arms + 1)))
            self.stability_ratio = 0.99
            self.min_exploration = int(100 * num_arms * (np.log(num_arms + 1)))
            self.arm_history = []

    def _update_confidence_radius(self) -> None:
        r"""Update the confidence radius using the latest failure probability.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def get_condorcet_winner(self) -> int:
        """Determine a Condorcet winner.
        """
        return argmax_set_1(np.sum(self.preference_estimate.get_mean_estimate_matrix().preferences, axis=1))

    def step(self) -> None:
        """Explore or Exploit
        """
        if self.is_finish_explore:
            self.exploit()
        else:
            self.explore()

    def exploit(self) -> None:
        """Run one step of exploitation.
        """
        self.wrapped_feedback.duel(self.best_arm, self.best_arm)

    def explore(self) -> None:
        """Run one round of exploration.
        """
        self.time_step += 1

        if self.time_step <= self.init_exploration_times:
            arm_c, arm_d = next(self.exploration_generator)
        else:
            # self._update_confidence_radius()
            # upper_confidence_bound_matrix = self.preference_estimate.get_upper_estimate_matrix()
            # upper_confidences = np.sum(upper_confidence_bound_matrix.preferences, axis=1)
            est = self.preference_estimate._cached_mean_estimate
            upper_confidences = np.sum(est, axis=1)
            arm_c, arm_d = argmax_set_2(upper_confidences)

            if self.is_exploit:
                # Update arm history and check stability
                self.arm_history.append(arm_c)
                if len(self.arm_history) > self.stability_window_size:
                    self.arm_history.pop(0)

                    if self.time_step > self.min_exploration:
                        arm_c_ratio = self.arm_history.count(arm_c) / len(self.arm_history)
                        if arm_c_ratio >= self.stability_ratio:
                            self.is_finish_explore = True
                            self.best_arm = self.get_condorcet_winner()
            
            if self.random_state.random() < self.greedy_ep:
                available_arms = list(range(len(upper_confidences)))
                available_arms.remove(arm_c)
                arm_d = self.random_state.choice(available_arms)
        
        user = self.choose_feedback_user()
        self.preference_estimate.enter_sample(
            arm_c, arm_d, self.wrapped_feedback.duel(arm_c, arm_d, user), user
        )
    