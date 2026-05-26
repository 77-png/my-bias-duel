"""An implementation of the Copeland Winners Relative Minimum Emprirical Divergence algorithm."""
import math
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.special import rel_entr

from duelpy.algorithms.algorithm import Algorithm
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.metrics import AverageCopelandRegret


class CwRmed(Algorithm):
    r"""Implement the CW-RMED algorithm.

    The goal of the algorithm is to minimize the :term:`Copeland regret`.
    It is assumed that there are no ties between arms, i.e. the
    probability of any arm winning against another is never :math:`\frac{1}{2}`.

    The bound on the expected regret is given as
    :math:`\frac{2N + \mathcal{o}(N)}{d_{KL}(\frac{1}{2} + \Delta, \frac{1}{2})}`.
    :math:`N` is the number of arms, :math:`d_{KL}` is the Bernoulli KL-divergence
    and :math:`\Delta` is the smallest absolute distance to :math:`\frac{1}{2}`
    in the probability of these arms dueling against each other.

    The CW-RMED algorithm is based on :cite:`komiyama2016copeland`. It
    uses Bernoulli KL-Divergence between the preference probabilities
    of arms and :math:`\frac{1}{2}`. Further, linear optimization is used
    on these arm pairs to select the optimal pairs for duels.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        Number of time steps to execute for.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    exploratory_constant
        Corresponds to :math:`\alpha` in :cite:`komiyama2016copeland`. Its value
        must be greater than or equal to 0.
        Default value is 3.0 which has been used in the experiments in
        :cite:`komiyama2016copeland`.
    regret_bound_constant
        Corresponds to :math:`\beta` in :cite:`komiyama2016copeland`. Its value
        must be greater than or equal to 0.
        Default value is 0.01 which has been used in the experiments in
        :cite:`komiyama2016copeland`.

    Attributes
    ----------
    wrapped_feedback
        The ``feedback_mechanism`` parameter with an added decorator. This
        feedback mechanism will raise an exception if a time horizon is given
        and a duel would exceed it. The exception is caught in the ``run``
        function.
    current_copeland_winner_pairs
        Corresponds to :math:`L_{NC}` in :cite:`komiyama2016copeland`.
    dueling_pairs
        Corresponds to :math:`L_C` in :cite:`komiyama2016copeland`.
    dueling_pairs_for_next_step
        Corresponds to :math:`L_N` in :cite:`komiyama2016copeland`.
    remaining_pairs_for_dueling
        Corresponds to :math:`L_R` in :cite:`komiyama2016copeland`.

    Raises
    ------
    ValueError
        Raised when the constants are given negative values.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> from duelpy.stats.metrics import AverageCopelandRegret
    >>> from duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, arms, random_state=random_state),
    ...     metrics={"copeland_regret": AverageCopelandRegret(preference_matrix)}
    ... )
    >>> cwrmed = CwRmed(
    ...     feedback_mechanism=feedback_mechanism,
    ...     time_horizon=100,
    ...     random_state=random_state
    ... )
    >>> cwrmed.run()
    >>> np.round(np.sum(feedback_mechanism.results["copeland_regret"]), 2)
    31.5
    >>> cwrmed.wrapped_feedback.duels_conducted
    100
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 3.0,
        regret_bound_constant: float = 0.01,
    ) -> None:
        super().__init__(
            feedback_mechanism,
            time_horizon,
        )
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        if exploratory_constant < 0 or regret_bound_constant < 0:
            raise ValueError("Value of constants must be greater than 0")
        self.exploratory_constant = exploratory_constant
        self.regret_bound_constant = regret_bound_constant

        # Referred as ``L_C`` in the paper
        self.dueling_pairs: Set[Tuple[int, int]] = set(
            self.wrapped_feedback.get_dueling_pair_combinations()
        )
        # Referred as ``L_R`` in the paper
        self.remaining_pairs_for_dueling: Set[
            Tuple[int, int]
        ] = self.dueling_pairs.copy()
        # Referred as ``L_N`` in the paper
        self.dueling_pairs_for_next_step: Set[Tuple[int, int]] = set()
        # Referred as ``L_{NC}`` in the paper
        self.current_copeland_winner_pairs: Set[Tuple[int, int]] = set()

        self.copeland_winner: int = 0
        self.preference_estimate = PreferenceEstimate(
            self.wrapped_feedback.get_num_arms()
        )

    def step(self) -> None:
        """Run one round of the algorithm."""
        try:
            self._conditional_sampling()
            self._update_dueling_pairs_for_next_round()
        except ValueError:
            pass
        self.dueling_pairs = self.dueling_pairs_for_next_step.copy()

        # Add ``(self.copeland_winner, self.copeland_winner)`` for exploration
        self.dueling_pairs |= {(self.copeland_winner, self.copeland_winner)}
        self.remaining_pairs_for_dueling = self.dueling_pairs.copy()
        self.dueling_pairs_for_next_step = set()

    def _update_dueling_pairs_for_next_round(self) -> None:
        """Sample, analyze and accordingly update the dueling pairs.

        Raises
        ------
        AlgorithmFinishedException
            When the number of duels match the time horizon. Raised by the
            ``duel`` method. The exception can be accessed by
            ``self.wrapped_feedback.exception_class``.
        """
        for (arm_i, arm_j) in self.dueling_pairs:
            self.preference_estimate.enter_sample(
                arm_i,
                arm_j,
                self.wrapped_feedback.duel(arm_i, arm_j),
            )
            self.current_copeland_winner_pairs = set()

            processed_winners = self._check_confidence()

            remaining_winners = (
                self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
                - processed_winners
            )

            if len(remaining_winners) > 0:
                self._update_current_copeland_winner_pairs_for(list(remaining_winners))

            self.remaining_pairs_for_dueling -= {(arm_i, arm_j)}

            for (arm_1, arm_2) in self.current_copeland_winner_pairs & (
                set(self.wrapped_feedback.get_dueling_pair_combinations())
                - self.remaining_pairs_for_dueling
            ):
                self.dueling_pairs_for_next_step |= {(arm_1, arm_2)}

    def _conditional_sampling(self) -> None:
        """Sample from the dueling pair combinations conditionally.

        Raises
        ------
        AlgorithmFinishedException
            When the number of duels match the time horizon. Raised by the
            ``duel`` method. The exception can be accessed by
            ``self.wrapped_feedback.exception_class``.
        """
        for (arm_i, arm_j) in self.wrapped_feedback.get_dueling_pair_combinations():
            if self.wrapped_feedback.duels_conducted > 1:
                if self.preference_estimate.get_num_samples(
                    arm_i, arm_j
                ) < self.exploratory_constant * math.sqrt(
                    math.log(self.wrapped_feedback.duels_conducted)
                ) or self.preference_estimate.get_mean_estimate(
                    arm_i, arm_j
                ) < self.regret_bound_constant / math.log(
                    math.log(self.wrapped_feedback.duels_conducted)
                ):
                    self.preference_estimate.enter_sample(
                        arm_i,
                        arm_j,
                        self.wrapped_feedback.duel(arm_i, arm_j),
                    )
            else:
                self.preference_estimate.enter_sample(
                    arm_i,
                    arm_j,
                    self.wrapped_feedback.duel(arm_i, arm_j),
                )

    # pylint: disable=too-many-branches
    def _check_confidence(self) -> Set[int]:
        """Check whether a copeland winner can be put for dueling.

        Returns
        -------
        Set[int]
            A set of copeland winners which were processed.
        """
        processed_winners = set()
        for (
            copeland_winner
        ) in self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners():
            condition_satisfied = True
            for arm_i in self.wrapped_feedback.get_arms():
                for arm_j in self.wrapped_feedback.get_arms():
                    if arm_i > arm_j:
                        kl_div = rel_entr(
                            1
                            - self.preference_estimate.get_mean_estimate(arm_i, arm_j),
                            0.5,
                        ) + rel_entr(
                            self.preference_estimate.get_mean_estimate(arm_i, arm_j),
                            0.5,
                        )
                        if (
                            self.preference_estimate.get_num_samples(arm_i, arm_j)
                            / math.log(self.wrapped_feedback.duels_conducted)
                            * kl_div
                            > 1.0
                        ):
                            condition_satisfied = False
                            break
                    if condition_satisfied is False:
                        break
                if condition_satisfied is False:
                    continue

            inferiors = (
                self.preference_estimate.get_mean_estimate_matrix().get_losers_against(
                    copeland_winner
                )
            )

            arms_for_emp_div: Set[Tuple[int, int]] = set()
            for arm in inferiors:
                arms_for_emp_div |= {(copeland_winner, arm)}

            for arm in self.wrapped_feedback.get_arms():
                if arm is not copeland_winner:
                    superiors = self.preference_estimate.get_mean_estimate_matrix().get_winners_against(
                        arm
                    )
                    for superior in superiors:
                        arms_for_emp_div |= {(arm, superior)}

            emp_div: List = list()
            for (arm_i, arm_j) in arms_for_emp_div:
                kl_div = rel_entr(
                    1 - self.preference_estimate.get_mean_estimate(arm_i, arm_j),
                    0.5,
                ) + rel_entr(
                    self.preference_estimate.get_mean_estimate(arm_i, arm_j),
                    0.5,
                )
                emp_div.append(
                    self.preference_estimate.get_num_samples(arm_i, arm_j)
                    / math.log(self.wrapped_feedback.duels_conducted)
                    * kl_div
                )

            emp_div.sort()

            constraint_size = len(
                self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
            )

            constrained_emp_div = emp_div[0:constraint_size]
            if sum(constrained_emp_div) > 1.0:
                self.current_copeland_winner_pairs |= {
                    (copeland_winner, copeland_winner)
                }
                processed_winners |= {copeland_winner}

        return processed_winners

    def _update_current_copeland_winner_pairs_for(
        self, remaining_winners: List
    ) -> None:
        """Update ``L_{NC}`` for this round."""
        self.copeland_winner = remaining_winners.pop()
        self._find_dueling_pairs()

    def _find_dueling_pairs(self) -> None:
        """Find eligible dueling pairs using the given copeland_winner with CWRMED algorithm."""
        self.current_copeland_winner_pairs |= {
            (self.copeland_winner, self.copeland_winner)
        }

        candidates = self.wrapped_feedback.get_arms()
        candidates.remove(self.copeland_winner)

        try:
            self._conduct_linear_optimization_for_candidates(candidates)
        except ValueError:
            pass

    def _conduct_linear_optimization_for_candidates(self, candidates: list) -> None:
        """Conduct linear optimization using scipy's linprog function.

        Parameters
        ----------
            candidates
                List of candidates.
        """
        regret = AverageCopelandRegret(
            self.preference_estimate.get_mean_estimate_matrix()
        )

        for candidate in candidates:
            superiors = np.array(
                self.preference_estimate.get_mean_estimate_matrix().get_winners_against(
                    candidate
                )
            )
            superiors = np.delete(
                superiors, np.argwhere(superiors == self.copeland_winner)
            )
            # ``c_j``
            costs = np.array([])
            kl_divergence = np.array([])
            comparison_array = np.array([])

            for superior in superiors:
                kl_div = rel_entr(
                    1 - self.preference_estimate.get_mean_estimate(superior, candidate),
                    0.5,
                ) + rel_entr(
                    self.preference_estimate.get_mean_estimate(superior, candidate), 0.5
                )
                kl_divergence = np.append(kl_divergence, kl_div)
                comparison_array = np.append(
                    comparison_array,
                    (
                        self.preference_estimate.get_num_samples(superior, candidate)
                        / math.log(self.wrapped_feedback.duels_conducted)
                    ),
                )
                costs = np.append(costs, regret(superior, candidate) / kl_div)

                # ``|S| - k = L_i_2 - L_i_1 + 1``
                constraint_array_size = (
                    len(
                        self.preference_estimate.get_mean_estimate_matrix().get_winners_against(
                            candidate
                        )
                    )
                    - len(
                        self.preference_estimate.get_mean_estimate_matrix().get_winners_against(
                            self.copeland_winner
                        )
                    )
                    + 1
                )
                constraint_array_size = abs(constraint_array_size)

                comparison_result = np.ndarray([])
                # Check for a non-trivial solution
                if np.size(costs) > 1:
                    constraint_matrix = np.ndarray([])
                    if np.size(costs) <= constraint_array_size:
                        constraint_matrix = np.array(
                            [np.full(np.size(costs), -1, dtype=int)]
                        )
                    else:
                        constraint_matrix = np.array(
                            [
                                np.append(
                                    np.full(constraint_array_size, -1, dtype=int),
                                    np.full(
                                        (np.size(costs) - constraint_array_size),
                                        0,
                                        dtype=int,
                                    ),
                                )
                            ]
                        )
                    comparison_result = np.greater(
                        np.multiply(
                            # Solve the linear optimization problem
                            linprog(
                                c=costs,
                                # Note: the inequality constraints must be in the form of <=
                                A_ub=constraint_matrix,
                                b_ub=np.array([-1]),
                            ).x,
                            kl_divergence,
                        ),
                        comparison_array,
                    )

                    for index in range(np.size(comparison_result)):
                        if comparison_result[index]:
                            self.current_copeland_winner_pairs |= {
                                (candidate, superiors[index])
                            }
