from typing import Optional
from typing import Union, List
import numpy as np

from duelpy.feedback.matrix_feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix
from duel_bias.model import UserBiasDueling

class BiasGenerator:
    """Generator basic class of evaluator bias
    """
    def __init__(self, seed):
        self.seed = seed

    def __iter__(self):
        self.generator = np.random.default_rng(self.seed)
        return self

    def __next__(self):
        raise NotImplementedError

class MatrixFeedbackBias(MatrixFeedback):
    """Compare two arms based on a preference matrix and the user bias

    Parameters
    ----------
    arms
        It represents a list of arms from preference matrix. If not provided, the method will create its own list of
        arms from preference matrix.
    preference_matrix
        A quadratic matrix where :math:`p[i, j]` specifies the probability that arm :math:`i`
        wins against arm :math:`j`. This implies :math:`p[j, i] = 1 - p[i, j]` and :math:`p[i, i] = 0.5`.
    user_bias
        A vector where :math:`eta[k]` specifies the judgement bias of user :math:`k`,
        i.e. the probability that, in the condition of arm :math:`i` wins against arm :math:`j`,
        user :math:`k` says that arm :math:`i` wins against arm :math:`j`.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    """

    def __init__(
        self,
        preference_matrix: Union[PreferenceMatrix, np.array],
        user_bias: Union[List[float], np.ndarray, List[BiasGenerator]],
        arms: Optional[list] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(preference_matrix, arms, random_state)
        self.user_bias = user_bias
        self.num_users = len(user_bias)
        self.choose_feedback_user = UserBiasDueling(self.num_users).choose_feedback_user

    def _sample_evaluator_bias(self, user_index: int):
        if isinstance(self.user_bias[user_index], BiasGenerator):
            return next(self.user_bias[user_index])
        else:
            return self.user_bias[user_index]

    def _compute_win_probability_with_bias(self, arm_i_index: int, arm_j_index: int, user_k_index: int = None) -> float:
        """Compute the probability that user k gives feedback that arm: math:`i` wins against arm: math:`j`
        """
        probability_i_wins = self.preference_matrix[arm_i_index][arm_j_index]

        if self.user_bias is None:
            user_bias = 1.0
        elif user_k_index is None:
            user_bias = self._sample_evaluator_bias(self.choose_feedback_user())
        else:
            user_bias = self._sample_evaluator_bias(user_k_index)

        return user_bias * probability_i_wins + (1.0 - user_bias) * (1.0 - probability_i_wins)

    def duel(self, arm_i_index: int, arm_j_index: int, user_k_index: int = None) -> bool:
        """Perform a duel between two arms based on a given probability matrix and user bias of i and j.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j
            If not provided, and self.user_bias is not None, then randomly select an user

        Returns
        -------
        bool
            True if ``arm_i_index`` wins.
        """
        return self.random_state.uniform() <= self._compute_win_probability_with_bias(arm_i_index, arm_j_index, user_k_index)

    def duel_repeatedly(self, arm_i_index: int, arm_j_index: int, duel_count: int, user_k_index: int = None) -> int:
        """Perform a duel between two arms multiple times based on a given probability matrix and user bias of i and j.

        If the ``duel_count`` is a large number in the range of :math:`500000`, use this method.
        The binomial distribution save the computations on multiple comparisons.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.
        user_k_index
            The user to judge the winner between arm_i and arm_j
        duel_count
            Number of duels that has to be performed.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        return self.random_state.binomial(duel_count, self._compute_win_probability_with_bias(arm_i_index, arm_j_index, user_k_index))
