"""Algorithm of the Binary Search Ranking."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from duelpy.algorithms.interfaces import CopelandRankingProducer
from duelpy.algorithms.interfaces import PacAlgorithm
from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.util.sorting import MergeSort
import duelpy.util.utility_functions as utility


class BinarySearchRanking(CopelandRankingProducer, PacAlgorithm):
    r"""Implementation of the Binary Search Ranking algorithm.

    This algorithm finds a :math:`\epsilon`-:term:`Copeland ranking` with probability :math:`1-\frac{1}{N}`, where :math:`N` is the number of arms.

    It is assumed that :term:`strong stochastic transitivity` and :term:`stochastic triangle inequality` hold.

    The sample complexity is bound by :math:`\mathcal{O}\left(\frac{N \log N}{\epsilon^2}\right)`.

    The algorithm divides arms into ordered bins, sorts these and finally merges them to get a ranking.
    To create the bins, first :math:`a = \frac{N}{(\log N)^3}` anchor arms are selected.
    These randomly selected anchor arms are ranked using a Rank-3 algorithm, which in our case is Mergesort.
    Arm comparisons are repeated until the preference is known with sufficient confidence.
    A bin is created between consecutive anchor arms, two additional bins are added before the first and after the last arm.
    The remaining arms are then sorted into the bins.
    The arms in the bins are divided into arms close to the anchor and the others. These other arms are between two anchor arms and not close enough to either one.
    They are sorted using the Rank-3 algorithm.
    The last step is merging the anchor arms, the arms close to each anchor arm and the arms between anchor arms in the correct order.

    The algorithm achieves a close to optimal sample complexity in theory, but there are large hidden constant factors, as can be seen in the example.
    The bias influences the sample complexity. Experiments with this implementation show for values greater than ``0.5`` a degradation of the probability of a correct result. For smaller values the sample complexity increases and the failure probability is stable at :math:`\frac{1}{N}`.

    Refer to the paper :cite:`falahatgar2017maximum` for more details.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        Optional, the number of arm comparisons to execute.
    epsilon
        The bias term referred to as :math:`\epsilon`.
    random_state
        Used for random choices in the algorithm.

    Attributes
    ----------
    random_state

    Examples
    --------
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> rank = BinarySearchRanking(feedback_mechanism, random_state=random_state)
    >>> rank.run()
    >>> rank.get_ranking()
    [2, 4, 3, 1, 0]
    >>> rank.wrapped_feedback.duels_conducted
    1786021
    """

    class Node:
        """Utility for implementation of Binary Search Tree.

        These nodes represent intervals on an array. The interval is split in the middle and the child nodes are assigned the left and right subinterval respectively.
        If the interval contains only one value, no child nodes are created.
        Refer to *Algorithm 9* in paper :cite:`falahatgar2017maximum` for more details.

        Attributes
        ----------
        low
            Points to the left boundary of the interval that the node represents.
        high
            Points to the right boundary of the interval that the node represents.
        mid
            Indicates where the interval is split for the child nodes.
        parent
            Points to the parent node in the tree.
        left
            Points to the left child node in the tree.
        right
            Points to the right child node in the tree.
        """

        def __init__(
            self,
            lower_bound: int,
            upper_bound: int,
            parent: Optional[BinarySearchRanking.Node] = None,
        ):
            self.low = lower_bound
            self.high = upper_bound
            self.parent = parent
            self.mid = math.ceil((lower_bound + upper_bound) / 2)
            self.left: Optional[BinarySearchRanking.Node] = None
            self.right: Optional[BinarySearchRanking.Node] = None
            if self.mid != self.high:
                self.left = BinarySearchRanking.Node(self.low, self.mid, self)
                self.right = BinarySearchRanking.Node(self.mid, self.high, self)

        def get_children(self) -> Tuple:
            """Return the left and right child of current node."""
            return self.left, self.right

        def get_parent(self) -> Optional[BinarySearchRanking.Node]:
            """Return the parent of the current node."""
            return self.parent

    loser_arm_dummy = -1
    winner_arm_dummy = -2

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        epsilon: float = 0.5,
        random_state: np.random.RandomState = None,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self._epsilon = epsilon
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self._final_result: Optional[List[int]] = None

    def exploration_finished(self) -> bool:
        """Return ``True`` if the ranking of arms has been created."""
        return self.get_ranking() is not None

    def get_ranking(self) -> Optional[List[int]]:
        """Return the ranked list of arms as decided by the algorithm."""
        return self._final_result

    def _binary_search(
        self,
        anchor_arms: List[int],
        visited_nodes_indices: List[int],
        search_arm: int,
        epsilon: float,
    ) -> int:
        """Implement BinarySearch Algorithm.

        Refer to Algorithm 10 for more details.

        Parameters
        ----------
        anchor_arms
            List of arms sorted using merge ranking.
        visited_nodes_indices
            List of nodes visited while traversing the tree.
        search_arm
            Element that is used to search node in ordered_node_list.
        epsilon
            Bias provided to the algorithm.

        Returns
        -------
        int
            Node from ordered_node_list.
        """
        start_index = 0
        end_index = len(visited_nodes_indices) - 1
        bucket_tolerance = 3 * epsilon

        while end_index - start_index > 0:
            interval_center = math.ceil((start_index + end_index) / 2)
            duel_count = int(
                10 * math.log(self.wrapped_feedback.get_num_arms() / epsilon ** 2)
            )
            mean_value = (
                self._duels_with_dummies(
                    search_arm,
                    anchor_arms[visited_nodes_indices[interval_center]],
                    duel_count,
                )
                / duel_count
            )

            if mean_value < 1 / 2 - bucket_tolerance:
                start_index = interval_center
            elif mean_value <= 1 / 2 + bucket_tolerance:
                # correct bucket found
                return visited_nodes_indices[interval_center]
            else:
                end_index = interval_center

        return visited_nodes_indices[end_index]

    def _create_ordered_anchors(self) -> Tuple[List[int], List[int]]:
        """Select anchors and rank them.

        Returns
        -------
        Tuple[List[int], List[int]]
            The ranked anchor arms and the remaining arms.
        """
        num_arms = self.wrapped_feedback.get_num_arms()
        arms = self.wrapped_feedback.get_arms()
        num_anchors = math.floor(num_arms / pow(math.log(num_arms), 3))
        anchor_arms = utility.pop_random(arms, self.random_state, amount=num_anchors)

        merge_sort = MergeSort(
            anchor_arms,
            lambda a1, a2: self._merge_compare_function(
                a1,
                a2,
                self._epsilon / 16,
                1 / pow(num_arms, 6),
            ),
            self.random_state,
        )
        while not merge_sort.is_finished():
            merge_sort.step()
        # for mypy, conversion from Optional[List[int]] to List[int]
        ranked_anchor_arms = merge_sort.get_result()
        assert ranked_anchor_arms is not None
        # add the weak arm which will lose against any arm and the strong arm which will beat all the arms.
        ranked_anchor_arms.insert(0, BinarySearchRanking.winner_arm_dummy)
        ranked_anchor_arms.append(BinarySearchRanking.loser_arm_dummy)

        return ranked_anchor_arms, arms

    def _sort_arms_in_bins(
        self, bins: Dict[int, List[int]], anchor_arms: List[int], arms: List[int]
    ) -> None:
        """Sort all remaining arms into their bins.

        Parameters
        ----------
        bins
            The bins for sorting.
        anchor_arms
            The arms marking the bin boundaries.
        arms
            The arms to sort in the bins.
        """
        # Refer to alpha in algorithm 8.
        root_node = BinarySearchRanking.Node(0, len(anchor_arms) - 1)

        # find anchor bucket for each arm
        for arm in arms:
            bin_index = self._find_bin(anchor_arms, arm, self._epsilon / 15, root_node)
            bins[bin_index].append(arm)

    def _sort_bin(
        self, arms_bin: List[int], current_anchor: int, next_anchor: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """Sort arms in a single bin.

        Parameters
        ----------
        arms_bin
            The arms in the bin to be sorted.
        current_anchor
            The anchor marking the beginning of the bin.
        next_anchor
            The anchor marking the end of the bin.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            The list of arms close to the first anchor, those in between (sorted) and those close to the next anchor
        """
        num_arms = self.wrapped_feedback.get_num_arms()
        duel_count = int(10 / pow(self._epsilon / 15, 2) * math.log(num_arms))
        tolerance = 6 * self._epsilon / 15
        close_to_current: List[int] = []
        between: List[int] = []
        close_to_next: List[int] = []
        for arm in arms_bin:
            # test if arm is close to current anchor
            mean_value = (
                self._duels_with_dummies(
                    arm,
                    current_anchor,
                    duel_count=duel_count,
                )
                / duel_count
            )
            if abs(mean_value - 1 / 2) <= tolerance:
                close_to_current.append(arm)
            else:
                # test if arm is close to next anchor
                mean_value = (
                    self._duels_with_dummies(
                        arm,
                        next_anchor,
                        duel_count=duel_count,
                    )
                    / duel_count
                )

                if abs(mean_value - 1 / 2) <= tolerance:
                    close_to_next.append(arm)
                else:
                    # arm is close to neither, treat as in between
                    between.append(arm)

        # sort arm in between anchors
        merge_sort = MergeSort(
            between,
            lambda a1, a2: self._merge_compare_function(
                a1,
                a2,
                self._epsilon / 15,
                num_arms ** -4,
            ),
            self.random_state,
        )

        while not merge_sort.is_finished():
            merge_sort.step()

        # for mypy, conversion from Optional[List[int]] to List[int]
        between_sorted = merge_sort.get_result()
        assert between_sorted is not None
        return close_to_current, between_sorted, close_to_next

    def explore(self) -> None:
        """Find the ranked set of arms.

        Implement the *Algorithm 4 (Binary Search Ranking)*.
        """
        anchor_arms, remaining_arms = self._create_ordered_anchors()

        bins: Dict[int, List[int]] = defaultdict(list)  # S_j

        self._sort_arms_in_bins(bins, anchor_arms, remaining_arms)

        self._final_result = []
        # needed to carry arms over to next loop iteration
        close_to_next_anchor: List[int] = []

        for bin_index, _ in enumerate(bins):
            close_to_current_anchor = (
                close_to_next_anchor  # start with those arms assigned in last iteration
            )
            (close_current, between_anchors, close_to_next_anchor,) = self._sort_bin(
                bins[bin_index], anchor_arms[bin_index], anchor_arms[bin_index + 1]
            )

            close_to_current_anchor.extend(close_current)

            if (
                anchor_arms[bin_index] != BinarySearchRanking.loser_arm_dummy
                and anchor_arms[bin_index] != BinarySearchRanking.winner_arm_dummy
            ):
                self._final_result.append(anchor_arms[bin_index])

            self._final_result.extend(close_to_current_anchor)
            self._final_result.extend(between_anchors)

    # pylint: disable=too-many-locals
    def _find_bin(
        self,
        anchor_arms: List[int],
        search_element: int,
        epsilon: float,
        root_node: BinarySearchRanking.Node,
    ) -> int:
        r"""Implement Interval Binary Search.

        This algorithm find which bin each element belongs to.
        According to the nodes in binary tree, it finds out the bin in which ``search_element`` should be placed.

        Refer to Algorithm 8 in the paper :cite:`falahatgar2017maximum`.

        Parameters
        ----------
        sorted_list
            List of elements sorted using Merge Ranking.
        search_element
            Element that is to be searched.
        epsilon
            Bias term. Refer to :math:`\epsilon` in the algorithm.
        root_node
            Refer to the root node of the binary tree.

        Returns
        -------
        int
            Value of node.
        """
        current_node = root_node
        visited_nodes_indices: Set[int] = set()  # Q in Algorithm 8.
        count = 0
        duel_count = int(10 / pow(epsilon, 2))

        for _ in range(int(30 * math.log(self.wrapped_feedback.get_num_arms()))):
            if (
                current_node.high - current_node.low > 1
            ):  # implies there are child nodes of this node.
                if current_node is None:
                    break
                visited_nodes_indices.add(current_node.low)
                visited_nodes_indices.add(current_node.mid)
                visited_nodes_indices.add(current_node.high)

                below_lower_bound = (
                    self._duels_with_dummies(
                        search_element,
                        anchor_arms[current_node.low],
                        duel_count,
                    )
                    / duel_count
                    > 0.5
                )
                above_upper_bound = (
                    self._duels_with_dummies(
                        anchor_arms[current_node.high],
                        search_element,
                        duel_count,
                    )
                    / duel_count
                    > 0.5
                )

                # arm violates either lower or upper bound
                if below_lower_bound or above_upper_bound:
                    # for mypy, conversion from Optional[Node] to Node
                    temp_node = current_node.get_parent()
                    # can never be None, since the root contains everything
                    assert temp_node is not None
                    current_node = temp_node

                else:

                    left_child, right_child = current_node.get_children()
                    arm_in_left_interval = (
                        self._duels_with_dummies(
                            search_element,
                            anchor_arms[current_node.mid],
                            duel_count,
                        )
                        / duel_count
                        > 0.5
                    )

                    current_node = left_child if arm_in_left_interval else right_child
            else:
                above_lower_bound = (
                    self._duels_with_dummies(
                        anchor_arms[current_node.low],
                        search_element,
                        duel_count,
                    )
                    / duel_count
                    > 0.5
                )
                below_upper_bound = (
                    self._duels_with_dummies(
                        search_element,
                        anchor_arms[current_node.high],
                        duel_count,
                    )
                    / duel_count
                    > 0.5
                )

                if above_lower_bound and below_upper_bound:
                    count += 1  # this increases our confidence that search arm belongs to current bin.
                else:
                    if count == 0:
                        # for mypy, conversion from Optional[Node] to Node
                        temp_node = (
                            current_node.get_parent()
                        )  # search arm doesn't belong to current bin.
                        assert temp_node is not None
                        current_node = temp_node
                    else:
                        count -= 1  # this decreases our confidence that search arm belongs to current bin.

        if count > 10 * math.log(self.wrapped_feedback.get_num_arms()):
            return (
                current_node.low
            )  # we are confident that search arm belong to the current bin.
        else:

            node_index_list = list(visited_nodes_indices)
            node_index_list.sort()
            return self._binary_search(
                anchor_arms, node_index_list, search_element, epsilon
            )

    def _duels_with_dummies(
        self, arm_i_index: int, arm_j_index: int, duel_count: int
    ) -> int:
        """Compare two arms repeatedly to estimate the preference.

        Compare two arms ``duel_count`` number of times to get an estimate of the preference.
        This wrapper allows for two dummy arms, a loser and a winner.

        Refer to *Algorithm 5 Compare2* in the paper :cite:`falahatgar2017maximum`.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.
        duel_count
            Number of duels to execute.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        if (
            arm_i_index == BinarySearchRanking.loser_arm_dummy
            or arm_j_index == BinarySearchRanking.winner_arm_dummy
        ):
            return 0
        if (
            arm_i_index == BinarySearchRanking.winner_arm_dummy
            or arm_j_index == BinarySearchRanking.loser_arm_dummy
        ):
            return duel_count

        wins = self.wrapped_feedback.duel_repeatedly(
            arm_i_index, arm_j_index, duel_count
        )

        return wins

    def _merge_compare_function(
        self,
        arm1: int,
        arm2: int,
        epsilon: float,
        failure_probability: float,
    ) -> int:
        r"""Compare two arms sufficiently often to guarantee an approximate result with high probability.

        Refer to *Algorithm 1* in :cite:`falahatgar2017maximum`.

        Parameters
        ----------
        arm1
            Index of the first arm.
        arm2
            Index of the second arm.
        epsilon
            The bias, called :math:`\epsilon` in the paper :cite:`falahatgar2017maximum`.
        failure_probability
            The probability of failure, called :math:`\delta` in the paper :cite:`falahatgar2017maximum`.

        Returns
        -------
        int
            1 if ``arm1`` beats ``arm2`` else -1.
        """

        def probability_scaling(num_samples: int) -> float:
            return 4 * num_samples ** 2

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability=failure_probability,
            probability_scaling_factor=probability_scaling,
        )
        estimate_probability = 0.5

        # Refer to Algorithm 1 in the paper :cite:`falahatgar2017maximum`.
        comparison_budget = (1 / (2 * pow(epsilon, 2))) * math.log(
            2 / failure_probability
        )

        # We cannot use the preference estimate here, because duels from the past should not be considered.
        iteration = 0
        wins = 0

        while (
            abs(estimate_probability - 0.5) <= confidence_radius(iteration) - epsilon
            and iteration <= comparison_budget
        ):
            if self.wrapped_feedback.duel(arm1, arm2):
                wins += 1
            iteration += 1
            estimate_probability = wins / iteration
        return 1 if estimate_probability > 0.5 else -1
