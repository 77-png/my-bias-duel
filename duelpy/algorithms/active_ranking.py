"""An implementation of Active Ranking algorithm."""
from typing import Collection
from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import GeneralizedRankingProducer
from duelpy.algorithms.interfaces import PacAlgorithm
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.util.utility_functions import pop_random


class ActiveRanking(GeneralizedRankingProducer, PacAlgorithm):
    r"""Implementation of the Active Ranking using pairwise comparisons.

    This algorithm finds the desired ranking of arms.

    It is assumed that a unique Borda ranking exists, which means there exists no pair of arms with the same Borda score.

    The sample complexity related to comparisons is :math:`\frac{n \log n}{(\tau_{k}-\tau_{k+1})^2}`.

    The algorithm maintains the disjoint sets(bins) which contains the arms. The user provide a list of border elements.
    Difference between two consecutive borders of the list specifies the size of bins. These bins are ranked in order
    from best group of arms to worst group of arms.

    The number of bins is referred by :math:`L` in the paper and the collections of bins with their sizes is, referred
    by :math:`\{k_l\}_{l=1}^{L}`.
    So if :math:`L=n` and :math:`(k_1,k_2,\dots,k_n) = (1,2,\dots,n)`, the partition :math:`\{S_l\}_{l=1}^{L}` will produce the
    full ranking over all the arms. But if :math:`L=2` and :math:`k_1=k`, then the set partition :math:`(S_1, S_2)`
    split :math:`[n]` into top k elements and its complement.

    For more details, please refer to the paper :cite:`heckel2019active`

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    failure_probability
        An upper bound on the acceptable probability to fail. Default to ``0.15`` as mentioned in paper :cite:`heckel2019active`.
    border_element_list
        Ascending list of elements specifying the border of bins. Difference between two consecutive borders specify the
        size of the partitions. If not provided, the algorithm will assume a full ranking has to be returned.
        Refer to :math:`k_l` in paper :cite:`heckel2019active`.

    Attributes
    ----------
    wrapped_feedback
        The ``feedback_mechanism`` parameter with an added decorator. This
        feedback mechanism will raise an exception if a time horizon is given
        and a duel would exceed it. The exception is caught in the ``run``
        function.


    Examples
    --------
    Find the ranking in this example:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ... [0.5, 0.3, 0.1, 0.1, 0.1],
    ... [0.7, 0.5, 0.3, 0.2, 0.2],
    ... [0.9, 0.7, 0.5, 0.8, 0.9],
    ... [0.9, 0.8, 0.2, 0.5, 0.2],
    ... [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix,
    ...                                       random_state=random_state)
    >>> active_rank = ActiveRanking(feedback_mechanism=feedback_mechanism, random_state=random_state, failure_probability=0.15, border_element_list=[1,2,3,4,5])
    >>> active_rank.run()
    >>> comparisons = active_rank.wrapped_feedback.duels_conducted
    >>> rank = active_rank.get_ranking()
    >>> rank
    [[2], [4], [3], [1], [0]]
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.15,
        border_element_list: Optional[List[int]] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon=time_horizon)
        self._remaining_arm: List[int] = self.wrapped_feedback.get_arms()
        self._random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        self.failure_probability = (
            failure_probability / self.wrapped_feedback.get_num_arms()
        )  # refer to equation 3.1 in paper :cite:`heckel2019active`.

        if border_element_list is None:
            border_element_list = list(
                range(1, self.wrapped_feedback.get_num_arms() + 1)
            )

        self._bins_count = border_element_list[-1]  # refers to :math:`L`.
        self._current_round = 0

        self._bins_list: List[List[int]] = [
            [] for _ in range(self._bins_count)
        ]  # refers to :math:`S_l`.

        self._bins_border_list = [-1] * len(border_element_list)

        self._validate_border_elements(
            border_element_list
        )  # throws error if the list of border elements is wrong.

        # refers to :math:`\{\tau_1, \tau_2, \dots, \tau_n\}`.
        self._estimated_score_arms: List[float] = [
            0.0
        ] * self.wrapped_feedback.get_num_arms()
        self._sorted_arm_list = (
            self.wrapped_feedback.get_arms()
        )  # initialize with random ordered list.

    def _get_updated_alpha(self) -> float:
        """Refer to equation 3.1 in paper :cite:`heckel2019active`."""
        beta = (
            np.log(1 / self.failure_probability)
            + 0.75 * np.log(np.log(1 / self.failure_probability))
            + 0.5 * np.log(1 + np.log(self._current_round / 2))
        )
        return np.sqrt(beta / (2 * self._current_round))

    def _validate_border_elements(self, bins_border_list: List[int]) -> None:
        """Return True if border_elements are in accepted order."""
        if bins_border_list[0] < 1:
            raise ValueError("The first border element has to be 1 or greater than 1.")
        for index, next_element in enumerate(bins_border_list):
            if next_element <= bins_border_list[0] and index > 0:
                raise ValueError(
                    "The border elements list should maintain ascending order."
                )
            # indexes the elements from 0.
            self._bins_border_list[index] = (
                bins_border_list[index] - 1
            )  # refers to :math:`\{\k_1, \k_2, \dots, \k_L\}`. As indexing start at 0, hence 1 subtracted.

    def _calculate_estimated_score(self) -> None:
        """Calculate the estimated score for each arm.

        For each selected arm, this method compares the arm against any random arm selected from the list of arms minus
        the arm selected.
        """
        for first_arm in self._remaining_arm:
            arms_copy = self.wrapped_feedback.get_arms()
            arms_copy.remove(first_arm)
            second_arm = pop_random(arms_copy, self._random_state)[0]
            result_duel = self.wrapped_feedback.duel(first_arm, second_arm)

            # refer to equation 3.2 in Algorithm 1.
            score_estimation = (
                (self._current_round - 1)
                * self._estimated_score_arms[first_arm]
                / self._current_round
            )

            self._estimated_score_arms.__setitem__(
                first_arm,
                (
                    score_estimation + 1 / self._current_round
                    if result_duel
                    else score_estimation
                ),
            )

    def _comparison_with_previous_border(self, bin_location: int, arm: int) -> bool:
        """Refer to equation 3.3a for more details in the paper :cite:`heckel2019active`.

        Return True if either this is the first bin or the estimated score of arm is less than the same of arm in the
        previous bin.
        """
        if self._bins_border_list[bin_location] < 0:
            return False  # this bin is already full.
        if self._bins_border_list[bin_location] == 0:
            return True  # arm is either in this bin or the latter bin.

        # refers to :math:`\hat{k}_(l-1)}`.
        previous_border = self._bins_border_list[bin_location - 1]
        current_border = self._bins_border_list[bin_location]
        if current_border == previous_border:
            return False
        return (
            self._estimated_score_arms[arm]
            < self._estimated_score_arms[previous_border]
            - 4 * self._get_updated_alpha()
        )

    def _comparison_with_next_border(self, bin_location: int, arm: int) -> bool:
        """Refer to equation 3.3b for more details in the paper :cite:`heckel2019active`.

        Return True if either the current bin index is equal to last bin or the estimated score of the arm is greater
        than the score of arm in the next bin.
        """
        if self._bins_border_list[bin_location] == len(self._remaining_arm) - 1:
            return True  # this is the location of border element of the last bin and hence there is no next border element.
        # print(self._current_round)
        next_border = (
            self._bins_border_list[bin_location] + 1
        )  # refer to :math:`\hat{k}_l + 1`

        return (
            self._estimated_score_arms[arm]
            > self._estimated_score_arms[next_border] + 4 * self._get_updated_alpha()
        )

    def _is_arm_in_bin(self, bin_location: int, arm: int) -> bool:
        """Return True if arm can be placed in the bin at location bin_location."""
        return self._comparison_with_previous_border(
            bin_location, arm
        ) and self._comparison_with_next_border(bin_location, arm)

    def explore(self) -> None:
        """Run one step of exploration."""
        self._current_round += 1
        self._calculate_estimated_score()

        # sorts the arms in descending order of estimated scores.
        self._remaining_arm.sort(key=lambda x: self._estimated_score_arms[x])

        for arm in self._remaining_arm:
            for bin_index in range(self._bins_count):
                if self._is_arm_in_bin(bin_index, arm):
                    self._bins_list[bin_index].append(
                        arm
                    )  # add arm to the :math:`\hat{S}_l`.
                    self._remaining_arm.remove(arm)
                    if bin_index == self._bins_count - 1:
                        self._bins_border_list[bin_index] = (
                            self._bins_border_list[bin_index] - 1
                        )
                        break
                    for bin_index_new in range(bin_index, self._bins_count):
                        self._bins_border_list[bin_index_new] = (
                            self._bins_border_list[bin_index_new] - 1
                        )
                    break

    def exploration_finished(self) -> bool:
        """Return True if the stopping condition of the algorithm has been satisfied."""
        return len(self._remaining_arm) == 0

    def get_top_k(self) -> Optional[Collection[int]]:
        """Return the top k elements."""
        return self._bins_list[0]

    def get_ranking(self) -> Optional[List[List[int]]]:
        """Return the list of ranked sets of k elements.

        Here, k is defined by the list of border elements given in input. It can be different for each set in the
        final list.
        """
        return self._bins_list
