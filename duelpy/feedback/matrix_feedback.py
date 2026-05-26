"""Gather feedback from a ground-truth preference matrix."""

from typing import Optional
from typing import Union

import numpy as np

from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats.preference_matrix import PreferenceMatrix


class MatrixFeedback(FeedbackMechanism):
    """Compare two arms based on a preference matrix.

    Parameters
    ----------
    arms
        It represents a list of arms from preference matrix. If not provided, the method will create its own list of
        arms from preference matrix.
    preference_matrix
        A quadratic matrix where :math:`p[i, j]` specifies the probability that arm :math:`i`
        wins against arm :math:`j`. This implies :math:`p[j, i] = 1 - p[i, j]` and :math:`p[i, i] = 0.5`.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    """

    def __init__(
        self,
        preference_matrix: Union[PreferenceMatrix, np.array],
        arms: Optional[list] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        if not isinstance(preference_matrix, PreferenceMatrix):
            preference_matrix = PreferenceMatrix(preference_matrix)
        if arms is None:
            arms = list(range(preference_matrix.get_num_arms()))
        else:
            if preference_matrix.get_num_arms() != len(arms):
                raise ValueError("Labels and matrix size mismatch")
        super().__init__(arms)
        self.preference_matrix = preference_matrix
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms based on a given probability matrix.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.

        Returns
        -------
        bool
            True if ``arm_i_index`` wins.
        """
        probability_i_wins = self.preference_matrix[arm_i_index][arm_j_index]
        i_wins = self.random_state.uniform() <= probability_i_wins
        return i_wins

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
    ) -> int:
        """Perform the duels between two arms multiple times.

        If the ``duel_count`` is a large number in the range of :math:`500000`, use this method.
        The binomial distribution save the computations on multiple comparisons.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.
        duel_count
            Number of duels that has to be performed.
        duel_limit
            Number of duels that the algorithm has a budget to allow. This is an Optional parameter.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        arm_i_wins = np.random.binomial(
            duel_count,
            self.preference_matrix.preferences[arm_i_index, arm_j_index],
        )

        return arm_i_wins
