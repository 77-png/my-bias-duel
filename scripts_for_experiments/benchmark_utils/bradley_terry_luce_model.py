import numpy as np
from numpy import ndarray
from typing import List, Union

from duelpy.feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix
from duelpy.util.utility_functions import argmax_set
from duel_bias.feedback import MatrixFeedbackBias
from duel_bias.feedback.matrix_feedback_bias import BiasGenerator


class BradleyTerryLuceModel(MatrixFeedback):
    r"""A feedback-mechanism based on the Bradley-Terry-Luce model.

    A preference matrix is generated from a :term:`Bradley-Terry-Luce distribution`.
    The probabilities are based on a non-negative 'skill' value for each arm.
    """

    def __init__(
        self,
        num_arms: int,
        random_state: Union[np.random.RandomState, np.random.default_rng],
        skill_vector: Union[List[float], ndarray] = None,
    ):
        if skill_vector is not None:
            if len(skill_vector) != num_arms:
                raise ValueError(
                    "The num_arms parameter needs to be equal to the length of the given skill_vector."
                )
            if any([x < 0 for x in skill_vector]):
                raise ValueError("The values in skill_vector must be non-negative.")
        else:
            skill_vector = random_state.uniform(size=num_arms)
        self._skill_vector = skill_vector

        preferences = np.full((num_arms, num_arms), 0.5)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx + 1, num_arms):
                relative_preference = skill_vector[first_arm_idx] / (
                    skill_vector[first_arm_idx] + skill_vector[second_arm_idx]
                )
                preferences[first_arm_idx][second_arm_idx] = relative_preference
                preferences[second_arm_idx][first_arm_idx] = 1 - relative_preference
        preference_matrix = PreferenceMatrix(preferences)

        super().__init__(preference_matrix=preference_matrix, random_state=random_state)

    def get_best_arms(self) -> List[int]:
        return argmax_set(self._skill_vector)

    def get_arbitrary_ranking(self) -> List[int]:
        return sorted(
            list(range(len(self._skill_vector))), key=lambda i: self._skill_vector[i], reverse=True
        )

    def test_ranking(self, ranking: List[int]) -> bool:
        current_skill = self._skill_vector[ranking[0]]
        for arm in ranking:
            if self._skill_vector[arm] > current_skill:
                return False
            current_skill = self._skill_vector[arm]
        return True
    
import numpy as np


import numpy as np
from duelpy.feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix
from duelpy.util.utility_functions import argmax_set


class BradleyTerryLuceModelBias(MatrixFeedback):
    """
    Group-aware biased dueling bandits environment.

    P(i≻j | user k) = σ( s_i − s_j + θ_k · (g_i − g_j) )

    IMPORTANT: The parent MatrixFeedback holds the UNBIASED skill matrix.
    Bias is applied dynamically in duel() per evaluator.
    Baseline algorithms that call duel(i,j) without a user index receive
    a random evaluator and therefore see correctly biased signal.

    Bug fix vs previous version: the parent preference matrix is now built
    from skill_vector only (no avg_theta baked in). The old version baked
    avg_theta into the parent matrix, causing baselines to see a different
    (doubly-biased) signal than MyGroup, making comparison unfair.
    """

    def __init__(
        self,
        num_arms: int,
        num_users: int,
        random_state,
        skill_vector=None,
        user_bias=None,
        group_membership=None,
        score_std: float = 1.0,
        reference_group: int = 1,
        max_duels: int = None,
        use_barp_model: bool = False,
    ):
        self.random_state = random_state
        self.num_arms = num_arms
        self.num_users = num_users

        self.skill_vector = (
            random_state.normal(0, score_std, num_arms)
            if skill_vector is None
            else np.array(skill_vector, dtype=float)
        )

        gm = (
            random_state.binomial(1, 0.5, num_arms)
            if group_membership is None
            else np.array(group_membership, dtype=int)
        )
        self.group_membership = gm
        self.reference_group = reference_group
        self.g = (gm == reference_group).astype(int)

        self.user_bias = (
            random_state.normal(0, 1.0, num_users)
            if user_bias is None
            else np.array(user_bias, dtype=float)
        )

        avg_theta = np.mean(self.user_bias)
        biased_matrix = np.full((num_arms, num_arms), 0.5)
        for i in range(num_arms):
            for j in range(num_arms):
                if i != j:
                    s_i_eff = self.skill_vector[i]
                    s_j_eff = self.skill_vector[j]
                    biased_matrix[i, j] = 1.0 / (1.0 + np.exp(-(s_i_eff - s_j_eff)))

        super().__init__(
            preference_matrix=PreferenceMatrix(biased_matrix),
            random_state=random_state,
        )
        self.duels = 0
        self.duel_history = []

    def duel(self, arm_i: int, arm_j: int, user: int = None) -> bool:
        """
        Biased duel. Returns True if arm_i wins.

        Parameters
        ----------
        arm_i, arm_j : int
            The two compared arms.
        user : int, optional
            Evaluator index. If None, an evaluator is sampled uniformly.

        Notes
        -----
        Every duel is recorded in ``self.duel_history`` for
        cumulative weak-regret evaluation.
        """
        if user is None:
            user = int(self.random_state.randint(0, self.num_users))

        # Record the actual pair selected by the algorithm.
        self.duel_history.append(
            (int(arm_i), int(arm_j), int(user))
        )

        theta = float(self.user_bias[int(user)])
        s_i = self.skill_vector[arm_i] + theta * self.g[arm_i]
        s_j = self.skill_vector[arm_j] + theta * self.g[arm_j]

        p = 1.0 / (1.0 + np.exp(-(s_i - s_j)))
        result = bool(self.random_state.rand() < p)

        self.duels += 1
        return result

    def sample(self, arm_i: int, arm_j: int) -> int:
        return arm_i if self.duel(arm_i, arm_j) else arm_j

    def sample_unbiased(self, arm_i: int, arm_j: int) -> int:
        diff = self.skill_vector[arm_i] - self.skill_vector[arm_j]
        p = 1.0 / (1.0 + np.exp(-diff))
        return arm_i if self.random_state.rand() < p else arm_j

    def get_best_arm(self) -> int:
        return int(np.argmax(self.skill_vector))

    def get_best_arms(self):
        return argmax_set(self.skill_vector)