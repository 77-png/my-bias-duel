"""Extension of biased evaluator to duelpy.stats.preference_estimate
"""
from typing import Optional
from types import SimpleNamespace
import numpy as np
from collections import defaultdict

from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate
from duelpy.stats.preference_matrix import PreferenceMatrix

import duel_bias.config.bias_estimation
import duel_bias.config.preference_estimation

class PreferenceEstimateBias(PreferenceEstimate):
    """Estimate preference with user bias

    An extension to the base class PreferenceEstimate
    """

    def __init__(
        self,
        num_arms: int,
        num_users: int,
        confidence_radius: ConfidenceRadius = TrivialConfidenceRadius(),
        assigned_user_bias: Optional[np.array] = None,
        init_times: Optional[int] = None,
        step_times: Optional[int] = None,
    ) -> None:
        super().__init__(num_arms, confidence_radius)
        self.num_users = num_users
        self.wins = np.tile(self.wins, (num_users, 1, 1))

        # mean estimation with user
        self._cached_mean_estimate = np.full((num_arms, num_arms), 0.5)
        self._aux_mean_estimate = np.full((num_arms, num_arms), 0.5)
        self._aux_mean_estimate_user = np.full((num_users, num_arms, num_arms), 0.5)

        # Initialization of user bias
        self._cached_user_bias = np.ones(num_users)

        # For testing the case that user bias is known.
        self._assigned_user_bias = assigned_user_bias

        # Read config
        self.config = SimpleNamespace(
            type_bias = duel_bias.config.bias_estimation.Config.type,
            type_pij = duel_bias.config.preference_estimation.Config.type
        )

        self._tmp_times = 0
        self._init_times = init_times if init_times is not None else 60 * num_arms * np.log(num_arms)
        self._step_times_bias = step_times if step_times is not None else num_arms # Update evaluator bias every n rounds
        self._improved_radius = True

        # Cached value for improved radius
        if self._improved_radius:
            self._num_samples = np.zeros((num_arms, num_arms))
            self._cached_unique_samples = np.array([0])
            self._cached_unique_samples_degrees = defaultdict(int)
            self._cached_unique_samples_degrees[0] = (num_arms * (num_arms - 1)) // 2 + num_arms
            self._cached_inverse_indices = np.zeros((num_arms, num_arms), dtype=int)
        else:
            self._num_samples = None
            self._cached_unique_samples = None
            self._cached_unique_samples_degrees = None
            self._cached_inverse_indices = None

        match self.config.type_pij:
            case 'weight-bias_inv':
                self._compute_biased_mean_preference = self._compute_biased_mean_preference_bias_inv_
            case 'weight-identity':
                self._compute_biased_mean_preference = self._compute_biased_mean_preference_id_
            case 'weight-identity-N':
                self._compute_biased_mean_preference = self._compute_biased_mean_preference_idn_
            case _:
                raise ValueError("Unknown config value for preference estimation.")

    def get_user_bias(self) -> np.array:
        """Return user bias
        """
        return self._cached_user_bias

    def enter_sample(
        self, first_arm_index: int, second_arm_index: int, first_won: bool, user_feedback: int = None
    ) -> None:
        """enter the result of a sampled duel.

        parameters
        ----------
        first_arm_index
            The index of the first arm in duel.
        second_arm_index
            The index of the second arm in duel.
        first_won
            Whether the first arm won the duel.
        user_feedback
            The index of user who gives the duel feedback.
        """

        if user_feedback is None:
            raise ValueError("An unexpected issue has arisen: user_feedback cannot be None")

        # Update history win array and num_samples cache
        if first_won:
            self.wins[user_feedback][first_arm_index][second_arm_index] += 1
        else:
            self.wins[user_feedback][second_arm_index][first_arm_index] += 1

        if self._improved_radius:
            self.update_cached_value(first_arm_index, second_arm_index)

        if first_arm_index == second_arm_index:
            return

        self._update_mean_estimate_user(first_arm_index, second_arm_index, first_won, user_feedback)
        self._update_aux_mean_estimate(first_arm_index, second_arm_index, first_won, user_feedback)
        self._update_user_bias()
        self._update_biased_mean_estimate()
        self._cached_radius = None

    def update_cached_value(self, first_arm_index, second_arm_index) -> None:
        if first_arm_index == second_arm_index:
            self._num_samples[first_arm_index][second_arm_index] += 1
        else:
            self._num_samples[first_arm_index][second_arm_index] += 1
            self._num_samples[second_arm_index][first_arm_index] += 1

        new_value = int(self._num_samples[first_arm_index][second_arm_index])
        old_value = int(new_value - 1)

        old_idx = self._cached_inverse_indices[first_arm_index][second_arm_index]

        # assert self._cached_unique_samples[old_idx] == old_value
        # assert self._cached_unique_samples_degrees[old_value] >= 1
        # if self._cached_unique_samples_degrees[old_value] == 1:
        #     # The last symmetry two items, which is going to be replaced with nex_idx
        #     assert np.sum(self._cached_inverse_indices == old_idx) == 2 if old_idx != 0 else 2 + self.num_arms

        if self._cached_unique_samples_degrees[old_value] == 1:
            new_idx = old_idx

            if new_value not in self._cached_unique_samples:
                # self._cached_unique_samples = np.delete(self._cached_unique_samples, old_idx)
                # self._cached_inverse_indices[self._cached_inverse_indices > old_idx] -= 1
                # self._cached_unique_samples = np.insert(self._cached_unique_samples, new_idx, values=new_value)
                # self._cached_inverse_indices[self._cached_inverse_indices >= old_idx] += 1
                self._cached_unique_samples[old_idx] = new_value
            else:
                self._cached_unique_samples = np.delete(self._cached_unique_samples, old_idx)
                self._cached_inverse_indices[self._cached_inverse_indices > old_idx] -= 1
        else:
            new_idx = old_idx + 1

            if new_value not in self._cached_unique_samples:
                self._cached_unique_samples = np.insert(self._cached_unique_samples, new_idx, values=new_value)
                self._cached_inverse_indices[self._cached_inverse_indices > old_idx] += 1

        self._cached_unique_samples_degrees[new_value] += 1
        self._cached_unique_samples_degrees[old_value] -= 1
        self._cached_inverse_indices[first_arm_index][second_arm_index] = new_idx
        self._cached_inverse_indices[second_arm_index][first_arm_index] = new_idx


    def get_radius_matrix(self) -> np.array:
        """Compute confidence radius for each pair of (arm i, arm j)
        This function will update the cached value of confidence radius and return it.
        """
        effective_radius = True

        if self._cached_radius is None:
            if self._improved_radius:
                num_samples = self._num_samples
                unique_num_samples, inverse_indices = self._cached_unique_samples, self._cached_inverse_indices
                # unique_num_samples, inverse_indices = np.unique(num_samples, return_inverse=True)
            else:
                num_samples = self.num_users * (self.wins + np.transpose(self.wins, axes=(0, 2, 1)))
                unique_num_samples, inverse_indices = np.unique(num_samples, return_inverse=True)

            unique_confidences = np.array(
                [self.confidence_radius(samples) for samples in unique_num_samples])
            full_confidences = unique_confidences[inverse_indices].reshape(num_samples.shape)

            if self._improved_radius:
                full_confidences = np.broadcast_to(full_confidences, (self.num_users, *full_confidences.shape))

            # Effective Confidence Radius or Theoretical one ?
            #   Theoretical Confidence Radius:
            #       The more user bias approaching 0.5,
            #       the less confidence for the estimated preference which is based on their biased feedback.
            #   Effective Confidence Radius:
            #       Although theoretical radius is reasonable to ensure a reliable solution,
            #       it is not effective in general (because of its overmuch exploration).
            if effective_radius:
                weight_bias = np.abs(2 * self._cached_user_bias - 1.0)
                weight_bias = weight_bias / np.sum(weight_bias)
            else:
                weight_bias = np.abs(2 * self._cached_user_bias - 1.0)
                weight_bias = weight_bias / np.sum(weight_bias ** 2)

            radius = np.tensordot(full_confidences, weight_bias, axes=(0, 0))
            np.fill_diagonal(radius, 0.0)
            self._cached_radius = radius

        return self._cached_radius

    def _update_biased_mean_estimate(self, sigma: float = 0.0) -> None:
        """Update estimated preference
        """
        self._cached_mean_estimate = self._compute_biased_mean_preference(sigma)

    def _update_user_bias(self):
        """Update User bias
        """
        self._tmp_times += 1
        init_phase = (self._tmp_times < self._init_times)

        if self._tmp_times % self._step_times_bias:
            return

        if self._assigned_user_bias is not None:
            self._cached_user_bias = self._assigned_user_bias
        else:
            # sigma = 0
            # sigma = self.num_arms
            sigma = 0.5 * self.num_arms ** 2

            match self.config.type_bias:
                case "optim-from-pref":
                    self._cached_user_bias = self._compute_user_bias(sigma=sigma)
                case "cond-from-pref":
                    self._cached_user_bias = self._compute_user_bias_cond()
                case "wcond-from-pref":
                    self._cached_user_bias = self._compute_user_bias_weight_cond()
                case "optim-from-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_from_mean_preference(sigma=sigma)
                case "cond-from-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_cond_from_mean_preference()
                case "wcond-from-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_weight_cond_from_mean_preference()
                case "optim-from-bias-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_from_biased_mean_preference(sigma=sigma)
                case "cond-from-bias-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_cond_from_biased_mean_preference()
                case "wcond-from-bias-mean-pref":
                    self._cached_user_bias = self._compute_user_bias_weight_cond_from_biased_mean_preference()
                case "optim-mix":
                    if init_phase:
                        self._cached_user_bias = self._compute_user_bias_from_mean_preference(sigma=sigma)
                        self._cached_user_bias = self._compute_user_bias_from_biased_mean_preference(sigma=sigma)
                    self._cached_user_bias = self._compute_user_bias_from_biased_mean_preference(sigma=sigma)
                case "cond-mix":
                    if init_phase:
                        self._cached_user_bias = self._compute_user_bias_cond_from_mean_preference()
                        self._cached_user_bias = self._compute_user_bias_cond_from_biased_mean_preference()
                    self._cached_user_bias = self._compute_user_bias_cond_from_biased_mean_preference()
                case "wcond-mix":
                    if init_phase:
                        self._cached_user_bias = self._compute_user_bias_weight_cond_from_mean_preference()
                        self._cached_user_bias = self._compute_user_bias_weight_cond_from_biased_mean_preference()
                    self._cached_user_bias = self._compute_user_bias_weight_cond_from_biased_mean_preference()
                case _:
                    raise ValueError("Unknown config value.")

    def _compute_biased_mean_preference(self, sigma: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def _compute_biased_mean_preference_bias_inv_(self, sigma: float = 0.0) -> np.ndarray:
        weight_bias = self.num_users * (2 * self._cached_user_bias - 1.0)
        temp_value = self._aux_mean_estimate_user + self._cached_user_bias[:, np.newaxis, np.newaxis] - 1.0
        return np.clip(np.sum(temp_value / weight_bias, axis=0), a_min=0.0, a_max=1.0)

    def _compute_biased_mean_preference_id_(self, sigma: float = 0.0) -> np.ndarray:
        weight_bias = 2 * self._cached_user_bias - 1.0
        if abs(sigma) < 1e-10:
            denominator = np.sum(weight_bias ** 2)
            numerator = np.tensordot(
                weight_bias,
                self._aux_mean_estimate_user + self._cached_user_bias[:, np.newaxis, np.newaxis] - 1.0,
                axes=(0, 0)
            )
        else:
            denominator = np.sum(weight_bias ** 2) + sigma
            numerator = np.tensordot(
                weight_bias,
                self._aux_mean_estimate_user + self._cached_user_bias[:, np.newaxis, np.newaxis] - 1.0,
                axes=(0, 0)
            ) + sigma * self._cached_mean_estimate
        return np.clip(numerator / denominator, a_min=0.0, a_max=1.0)

    def _compute_biased_mean_preference_idn_(self, sigma: float = 0.0) -> np.ndarray:
        num_feedback = np.sum(self.wins, axis=(1, 2))
        weight_bias = 2 * self._cached_user_bias - 1.0
        if abs(sigma) < 1e-10:
            denominator = np.sum(weight_bias ** 2 * num_feedback)
            numerator = np.tensordot(
                weight_bias * num_feedback,
                self._aux_mean_estimate_user + self._cached_user_bias[:, np.newaxis, np.newaxis] - 1.0,
                axes=(0, 0)
            )
        else:
            denominator = np.sum(weight_bias ** 2 * num_feedback) + sigma
            numerator = np.tensordot(
                weight_bias * num_feedback,
                self._aux_mean_estimate_user + self._cached_user_bias[:, np.newaxis, np.newaxis] - 1.0,
                axes=(0, 0)
            ) + sigma * self._cached_mean_estimate
        return np.clip(numerator / denominator, a_min=0.0, a_max=1.0)

    def _update_mean_estimate_user(
        self, first_arm_index: int, second_arm_index: int, first_won: bool, user_feedback: int = None
    ):
        samples = self.get_num_samples(first_arm_index, second_arm_index, user_feedback)
        prev = self._aux_mean_estimate_user[user_feedback][first_arm_index][second_arm_index]
        win_indicator = 1 if first_won else 0
        new_mean_user = prev + (win_indicator - prev) / samples

        self._aux_mean_estimate_user[user_feedback][first_arm_index][second_arm_index] = new_mean_user
        self._aux_mean_estimate_user[user_feedback][second_arm_index][first_arm_index] = 1.0 - new_mean_user

    def _update_aux_mean_estimate(
        self, first_arm_index: int, second_arm_index: int, first_won: bool, user_feedback: int = None
    ):
        samples = self.get_num_samples(first_arm_index, second_arm_index)
        prev = self._aux_mean_estimate[first_arm_index][second_arm_index]
        win_indicator = 1 if first_won else 0
        new_mean = prev + (win_indicator - prev) / samples

        self._aux_mean_estimate[first_arm_index][second_arm_index] = new_mean
        self._aux_mean_estimate[second_arm_index][first_arm_index] = 1.0 - new_mean

    # Compute user bias based on condition expression
    def _compute_user_bias_cond(self):
        mean_estimate = self._cached_mean_estimate
        return self._compute_user_bias_cond_from_estimate(mean_estimate)

    def _compute_user_bias_cond_from_mean_preference(self):
        mean_estimate = self._aux_mean_estimate
        return self._compute_user_bias_cond_from_estimate(mean_estimate)

    def _compute_user_bias_cond_from_biased_mean_preference(self):
        mean_estimate = self._compute_biased_mean_preference()
        return self._compute_user_bias_cond_from_estimate(mean_estimate)

    # Compute user bias based on weighted condition expression
    def _compute_user_bias_weight_cond(self):
        mean_estimate = self._cached_mean_estimate
        return self._compute_user_bias_weight_cond_from_estimate(mean_estimate)

    def _compute_user_bias_weight_cond_from_mean_preference(self):
        mean_estimate = self._aux_mean_estimate
        return self._compute_user_bias_weight_cond_from_estimate(mean_estimate)

    def _compute_user_bias_weight_cond_from_biased_mean_preference(self):
        mean_estimate = self._compute_biased_mean_preference()
        return self._compute_user_bias_weight_cond_from_estimate(mean_estimate)

    # Compute user bias based on optimization
    def _compute_user_bias(self, sigma: float = 1.0):
        """Compute user bias based on conditional probability equation.
        """
        mean_estimate = self._cached_mean_estimate
        return self._compute_user_bias_from_estimate(mean_estimate, sigma)

    def _compute_user_bias_from_mean_preference(self, sigma: float = 1.0):
        """Compute user bias based on conditional probability equation (and aux preference).
        """
        mean_estimate = self._aux_mean_estimate
        return self._compute_user_bias_from_estimate(mean_estimate, sigma)

    def _compute_user_bias_from_biased_mean_preference(self, sigma: float = 1.0):
        mean_estimate = self._compute_biased_mean_preference()
        return self._compute_user_bias_from_estimate(mean_estimate, sigma)

    # Utils
    def _compute_user_bias_cond_from_estimate(self, mean_estimate):
        """Compute user bias based on conditional probability expression.
        """
        superior = self.get_preference_ij(mean_estimate)
        superior_users = np.array([
            np.logical_and(superior, self.get_preference_ij(self._aux_mean_estimate_user[i]))
            for i in range(self.num_users) 
        ])
        num_superior = np.sum(superior_users, axis=1)
        total_superior = np.sum(superior)
        return num_superior / total_superior

    def _compute_user_bias_weight_cond_from_estimate(self, mean_estimate):
        """Compute user bias based on conditional probability expression (weighted formulation).
        """
        superior_weight = self.get_preference_weight_ij(mean_estimate)
        weighted_sums = np.array([
            superior_weight[self.get_preference_ij(self._aux_mean_estimate_user[i])].sum()
            for i in range(self.num_users)
        ])
        total_weight = np.sum(superior_weight)
        return weighted_sums / total_weight

    def _compute_user_bias_from_estimate(self, mean_estimate, sigma):
        num_feedback = np.sum(self.wins, axis=0)
        num_feedback = num_feedback + num_feedback.T
        weight_preference = 2 * mean_estimate - 1.0

        if abs(sigma) < 1e-10:
            denominator = np.sum(weight_preference**2 * num_feedback)
            numerator = np.einsum(
                'ij,kij->k',
                weight_preference * num_feedback,
                self._aux_mean_estimate_user + mean_estimate[np.newaxis, :, :] - 1.0
            )
        else:
            denominator = np.sum(weight_preference**2 * num_feedback) + sigma
            numerator = np.einsum(
                'ij,kij->k',
                weight_preference * num_feedback,
                self._aux_mean_estimate_user + mean_estimate[np.newaxis, :, :] - 1.0
            ) + sigma * self._cached_user_bias
        return np.clip(numerator / denominator, a_min=0.0, a_max=1.0)

    @staticmethod
    def get_preference_ij(preference, superior_gap: float = 0.05) -> np.ndarray:
        return preference.flatten(order='C') > 0.5 * (1.0 - superior_gap)

    @staticmethod
    def get_preference_weight_ij(preference, superior_gap: float = 0.05) -> np.ndarray:
        return np.maximum(2.0 * preference.flatten(order='C') - 1.0 + superior_gap, 0.0)

    def get_num_samples(self, first_arm_index: int, second_arm_index: int, user_index: int = None) -> int:
        """Get the number of times a duel between first_arm and second_arms was sampled.
        """
        if user_index is None:
            return np.sum(
                self.wins[:, first_arm_index, second_arm_index]
                + self.wins[:, second_arm_index, first_arm_index]
            )
        else:
            return np.sum(
                self.wins[user_index, first_arm_index, second_arm_index]
                + self.wins[user_index, second_arm_index, first_arm_index]
            )

    def sample_preference_matrix(
        self, random_state: np.random.RandomState
    ) -> PreferenceMatrix:
        """Sample a preference matrix based on a Beta distribution.
        """
        beta_a = np.sum(self.wins, axis=0) + 1
        beta_b = beta_a.T
        upper_triangle_preferences = random_state.beta(beta_a, beta_b)
        return PreferenceMatrix.from_upper_triangle(upper_triangle_preferences)

    def enter_sample_unbias(
        self, first_arm_index: int, second_arm_index: int, first_won: bool, user_feedback: int = None
    ) -> None:
        """Enter the result of a sampled duel without considering user bias.

        Compared to
        >>> self.enter_sample(...)
        this function provides an unbiased version for comparison.

        Parameters
        ----------
        first_arm_index
            The index of the first arm in duel.
        second_arm_index
            The index of the second arm in duel.
        first_won
            Whether the first arm won the duel.
        user_feedback
            The index of user who gives the duel feedback.
        """

        if user_feedback is None:
            raise ValueError("An unexpected issue has arisen: user_feedback cannot be None")

        # Update history win array and num_samples cache
        if first_won:
            self.wins[user_feedback][first_arm_index][second_arm_index] += 1
        else:
            self.wins[user_feedback][second_arm_index][first_arm_index] += 1

        if self._improved_radius:
            self.update_cached_value(first_arm_index, second_arm_index)

        if first_arm_index == second_arm_index:
            return

        samples = self.get_num_samples(first_arm_index, second_arm_index)
        prev = self._cached_mean_estimate[first_arm_index][second_arm_index]
        win_indicator = 1 if first_won else 0
        new_mean = prev + (win_indicator - prev) / samples

        self._cached_mean_estimate[first_arm_index][second_arm_index] = new_mean
        self._cached_mean_estimate[second_arm_index][first_arm_index] = 1.0 - new_mean
        self._cached_radius = None

    def get_radius_matrix_unbias(self) -> np.array:
        """Compute confidence radius for each pair of (arm i, arm j) without considering user bias.

        Compared to
        >>> self.get_radius_matrix()
        this function provides an unbiased version of radius for comparison.
        """
        if self._cached_radius is None:
            num_samples = np.sum(self.wins + np.transpose(self.wins, axes=(0, 2, 1)), axis=0)
            unique_num_samples, inverse_indices = np.unique(
                num_samples, return_inverse=True
            )
            unique_confidences = np.array(
                [self.confidence_radius(samples) for samples in unique_num_samples]
            )
            full_confidences = unique_confidences[inverse_indices].reshape(
                num_samples.shape
            )
            np.fill_diagonal(full_confidences, 0.0)
            self._cached_radius = full_confidences
        return self._cached_radius