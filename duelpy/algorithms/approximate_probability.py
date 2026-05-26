"""Implementation of the Approximate probability algorithm."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import PreferenceMatrixProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.preference_matrix import PreferenceMatrix


class ApproximateProbability(PreferenceMatrixProducer):
    r"""Implementation of the Approximate probability algorithm.

    The goal is to approximate the pairwise preference matrix between all arms.

    The algorithm assumes a :term:`total order` over the existing arms and that :term:`strong stochastic
    transitivity` and :term:`stochastic triangle inequality` hold. Additionally, a :math:`\frac{\epsilon}{8}`-approximate ranking over the arms has to be provided.

    The bound on the expected regret is given as :math:`\mathcal{O}\left(\frac{N\min\left\{N,\frac{1}{\epsilon}\right\}}{\epsilon^2}\right)`,
    where :math:`N` is the number of arms and :math:`\epsilon` is the targeted
    estimation accuracy.

    The approximate probability algorithm is based on *Algorithm 5* in :cite:`falahatgar2018limits`.
    It's an (:math:`\epsilon, \delta`)-:term:`PAC` algorithm with :math:`\delta = \frac{1}{N^2}`
    where :math:`N` is the number of arms.

    The algorithm takes an ordered set of arms and approximates all pairwise probabilities to
    an accuracy of :math:`\epsilon`. Note that in this implementation a ranking is defined as ordered from best to worst, whereas in :cite:`falahatgar2018limits`, this is reversed. Probabilities are calculated starting with the worst arm against all others and then iterating down the ranking order. The result is guaranteed to be consistent with the ranking.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    epsilon
        The optimality of the winning arm. Corresponds to :math:`\epsilon` in :cite:`falahatgar2018limits`.
        Default value is ``0.05``, which has been used in the experiments in :cite:`falahatgar2018limits`.
    order_arms
        A :math:`\frac{\epsilon}{8}` ranking over the arms, ordered from best to worst.

    Attributes
    ----------
    wrapped_feedback
        The ``feedback_mechanism`` parameter with an added decorator. This
        feedback mechanism will raise an exception if a time horizon is given
        and a duel would exceed it. The exception is caught in the ``run``
        function.
    tournament_arms
        The arms that are still in the tournament.
    estimate_pairwise_probability
    epsilon
    comparison_arm
        Iterate the number of comparisons of a specific arm.
    order_arms

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.9, 0.9],
    ...     [0.1, 0.5, 0.7],
    ...     [0.1, 0.3, 0.5]
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=np.random.RandomState(100))
    >>> test_object = ApproximateProbability(feedback_mechanism, epsilon=0.05, order_arms=[0, 1, 2])
    >>> test_object.run()
    >>> test_object.get_preference_matrix()
    array([[0.5, 0.9, 0.9],
    ...    [0.1, 0.5, 0.7],
    ...    [0.1, 0.3, 0.5]])
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        order_arms: List[int],
        epsilon: float = 0.05,
        time_horizon: Optional[int] = None,
    ):
        self.tournament_arms = feedback_mechanism.get_num_arms()
        self.comparison_arm = self.tournament_arms - 1
        super().__init__(feedback_mechanism, time_horizon)
        self.epsilon = epsilon
        self.order_arms = order_arms
        self._estimate_pairwise_probability: list = np.zeros(
            (self.tournament_arms, self.tournament_arms)
        )

        self.preference_estimate = PreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms()
        )

    def estimate_probabilities_against_worst_arm(self) -> None:
        """Run one step of comparison.

        The last ranked and the other arms are dueled repeatedly, determining their preference probabilities.
        """
        worst_arm = self.order_arms[-1]
        self._estimate_pairwise_probability[worst_arm][worst_arm] = 0.5
        for rank_index in range(self.tournament_arms - 2, -1, -1):
            other_arm = self.order_arms[rank_index]
            preceding_arm = self.order_arms[rank_index + 1]
            self._estimate_pairwise_probability[worst_arm][
                other_arm
            ] = self.duel_repeatedly(worst_arm, other_arm)
            self._estimate_pairwise_probability[other_arm][worst_arm] = (
                1 - self._estimate_pairwise_probability[worst_arm][other_arm]
            )

            if (
                self._estimate_pairwise_probability[other_arm][worst_arm]
                < self._estimate_pairwise_probability[preceding_arm][worst_arm]
            ):
                self._estimate_pairwise_probability[other_arm][
                    worst_arm
                ] = self._estimate_pairwise_probability[preceding_arm][worst_arm]
                self._estimate_pairwise_probability[worst_arm][other_arm] = (
                    1 - self._estimate_pairwise_probability[other_arm][worst_arm]
                )

    def estimate_pairwise_probabilities(self) -> None:
        """Run second step of comparison.

        It compares arm :math:`i` and arm :math:`j` multiple times and estimates the
        pairwise probability.
        """
        fixed_arm = self.order_arms[self.comparison_arm]
        preceding_fixed_arm = self.order_arms[self.comparison_arm + 1]
        self._estimate_pairwise_probability[fixed_arm][fixed_arm] = 0.5
        for rank_index in range(self.tournament_arms - 2, self.comparison_arm - 1, -1):
            other_arm = self.order_arms[rank_index]
            preceding_arm = self.order_arms[rank_index + 1]
            if (
                self._estimate_pairwise_probability[preceding_arm][fixed_arm]
                == self._estimate_pairwise_probability[other_arm][preceding_fixed_arm]
            ):

                self._estimate_pairwise_probability[other_arm][
                    fixed_arm
                ] = self._estimate_pairwise_probability[preceding_arm][other_arm]
                self._estimate_pairwise_probability[fixed_arm][other_arm] = (
                    1 - self._estimate_pairwise_probability[other_arm][fixed_arm]
                )
            else:

                self._estimate_pairwise_probability[other_arm][
                    fixed_arm
                ] = self.duel_repeatedly(other_arm, fixed_arm)
                self._estimate_pairwise_probability[fixed_arm][other_arm] = (
                    1 - self._estimate_pairwise_probability[other_arm][fixed_arm]
                )

    def duel_repeatedly(self, arm_i: int, arm_j: int) -> float:
        """Determine the preferred arm by repeated comparison.

        It calculates the number of times arm :math:`i` won against other arms in the set,
        and return the estimate pairwise probability.
        """
        compare_range = (int)(
            (16 / self.epsilon ** 2) * np.log(self.tournament_arms ** 4)
        )
        wins_i = 0
        for _ in range(compare_range):
            if self.wrapped_feedback.duel(arm_i, arm_j):
                wins_i += 1

        # approximate_probability corresponds to \hat\tilde p and is the estimated
        # win-fraction rounded to the nearest multiple of epsilon
        return np.round(wins_i / compare_range / self.epsilon) * self.epsilon

    def step(self) -> None:
        """Take multiple samples per step in the algorithm."""
        if self.comparison_arm == self.tournament_arms - 1:
            self.estimate_probabilities_against_worst_arm()
        else:
            self.estimate_pairwise_probabilities()

        self.comparison_arm -= 1

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        If the comparison arm is greater than tournament arms then it will terminate.
        """
        return self.comparison_arm < 0

    def get_preference_matrix(self) -> Optional[PreferenceMatrix]:
        """Return the computed preference matrix if it is ready.

        Returns
        -------
        Optional[PreferenceMatrix]
            The estimated pairwise preference matrix or ``None`` if the result
            is not ready.
        """
        return (
            PreferenceMatrix(self._estimate_pairwise_probability)
            if self.is_finished()
            else None
        )
