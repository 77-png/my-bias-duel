"""An implementation of the Borda ranking algorithm."""

import math
from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import BordaRankingProducer
from duelpy.algorithms.interfaces import PacAlgorithm
from duelpy.feedback import FeedbackMechanism


class BordaRanking(BordaRankingProducer, PacAlgorithm):
    r"""Implementation of the Borda ranking algorithm.

    The goal is to find a :math:`\epsilon`-:term:`Borda ranking` .

    It is assumed that :term:`strong stochastic transitivity` holds.

    The bound on the sample complexity is given as :math:`\mathcal{O}\left(\frac{2N}{\ln\epsilon^2}\log \frac{2N}{\delta}\right)`.

    The :term:`Borda ranking` algorithm is based on *Algorithm 9* in :cite:`falahatgar2017maxing`.
    It is an (:math:`\epsilon`, :math:`\delta`)-:term:`PAC` algorithm.

    The algorithm first computes the estimated :term:`Borda score` for all the arms in the set with an
    approximation error of :math:`\frac{\epsilon}{2}` with confidence at least :math:`1 − \frac{\delta}{N}`,
    and the value of these estimated :term:`Borda scores<Borda score>` will be use to rank the members of set.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
         The number of steps that the algorithm is supposed to be run.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    epsilon
        The optimality of the winning arm. Corresponds to :math:`\epsilon` in :cite:`falahatgar2017maxing`.
        Default value is ``0.05`` which has been used in the experiments in :cite:`falahatgar2017maxing`.
    failure_probability
        The probability that the result is not correct. Corresponds to :math:`\delta` in
        :cite:`falahatgar2017maxing`. Default value is ``0.1`` which has been used in the experiments in
        :cite:`falahatgar2017maxing`.

    Attributes
    ----------
    random_state
    epsilon
    failure_probability

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state=np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> test_object = BordaRanking(feedback_mechanism, epsilon=0.05, failure_probability=0.1)
    >>> test_object.run()
    >>> test_object.get_ranking()
    [2, 1, 0]
    >>> test_object.wrapped_feedback.duels_conducted
    9828
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        epsilon: float = 0.05,
        failure_probability: float = 0.1,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        self.epsilon = epsilon
        self.failure_probability = failure_probability
        self._current_arm = 0
        self._estimate_borda_scores = np.zeros(self.wrapped_feedback.get_num_arms())

    def exploration_finished(self) -> bool:
        """Determine if the actual algorithm has terminated.

        The algorithm terminates if all :term:`Borda scores<Borda score>` are estimated.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        return self._current_arm >= self.wrapped_feedback.get_num_arms()

    def explore(self) -> None:
        """Run one step of exploration."""
        wins = 0
        num_arms = self.wrapped_feedback.get_num_arms()
        comparison_budget = math.ceil(
            2 / self.epsilon ** 2 * math.log(2 * num_arms / self.failure_probability)
        )
        for _ in range(comparison_budget):
            random_arm = self.random_state.randint(num_arms)
            if self.wrapped_feedback.duel(self._current_arm, random_arm):
                wins += 1

        estimated_score = wins / num_arms

        self._estimate_borda_scores[self._current_arm] = estimated_score
        self._current_arm += 1

    def get_ranking(self) -> Optional[List[int]]:
        """Return the estimated Borda ranking.

        Returns
        -------
        Optional[List]
            The estimated ranking.
        """
        if self.exploration_finished():
            return list(np.argsort(-self._estimate_borda_scores))
        else:
            return None
