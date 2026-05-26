"""Implementations of verification based solution algorithms."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.algorithms.interfaces import PacAlgorithm
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius


class VerificationBasedCondorcet(CondorcetProducer, PacAlgorithm):
    r"""Condorcet variant of the verification based algorithm.

    This algorithm finds the :term:`Condorcet winner` with a given failure probability.

    It assumes a :term:`Condorcet winner` exists.

    The sample complexity of the algorithm is too complex to explain it here. It is given as *Theorem 5* in :cite:`karnin2016verification`.

    The algorithm executes multiple rounds of exploration followed by verification. The exploration step finds a possible :term:`Condorcet winner`, which the verification step tests. If the verification fails, the process is repeated.
    The two parameters trade off failure probability with sample complexity. A lower ``failure_probability`` (:math:`\delta`) means the verification takes longer, whereas a higher ``explore_failure_probability`` (:math:`\kappa`) means more explore-verify rounds may be executed.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        Optional, the number of time steps to execute for. If not provided, the algorithm terminates once a Condorcet winner is found.
    failure_probability
        The probability of returning an incorrect result. Referred to as :math:`\delta` in the paper.
    explore_failure_probability
        Determines the probability of failure in the explore step. Referred to as :math:`\kappa` in the paper. If not set, the default :math:`\min\left\{\frac{1}{3},\log\left(\frac{1}{\delta}\right)\right\}` is used, as described in `Corollary 4` of the paper.
    random_state
        Optional, used for random tie breaking. If not provided, a random state is created.

    Attributes
    ----------
    failure_probability
    explore_failure_probability
    round
        The current exploration-verification round index.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, arms, random_state)
    >>> vbc = VerificationBasedCondorcet(feedback_mechanism=feedback_mechanism, failure_probability=0.1, explore_failure_probability=0.5)
    >>> vbc.run()

    The best arm in this case is the last arm (index 2)

    >>> vbc.wrapped_feedback.duels_conducted
    3254
    >>> vbc.get_condorcet_winner()
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        failure_probability: float = 0.1,
        explore_failure_probability: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
        )
        self.failure_probability = failure_probability
        # Refer to the paragraph below Corollary 4 in the paper
        self.explore_failure_probability: float = (
            min(1 / 3, 1 / -np.log(failure_probability))
            if explore_failure_probability is None
            else explore_failure_probability
        )
        self.round = 1
        self._exploring = True
        self._explorer = VerificationBasedCondorcet.CondorcetExplorer(
            feedback_mechanism=self.wrapped_feedback,
            failure_probability=self.explore_failure_probability,
            random_state=random_state,
        )
        self._verifier = VerificationBasedCondorcet.CondorcetVerifier(
            feedback_mechanism=self.wrapped_feedback,
            best_arm=0,
            adversaries=self.wrapped_feedback.get_num_arms() * [0],
            failure_probability=self.failure_probability / 2 * 2 ** self.round,
        )  # dummy value, the verifier cannot be defined until the exploration is completed

    def _init_verifier(self) -> "VerificationBasedCondorcet.CondorcetVerifier":
        """Initialize the verifier object."""
        best_arm = self._explorer.get_best_arm()
        adversaries = self._explorer.get_adversaries()
        assert best_arm is not None
        assert adversaries is not None
        return VerificationBasedCondorcet.CondorcetVerifier(
            feedback_mechanism=self.wrapped_feedback,
            best_arm=best_arm,
            adversaries=adversaries,
            failure_probability=self.failure_probability / 2 * 2 ** self.round,
        )

    def explore(self) -> None:
        """Advance the algorithm by one step.

        This executes one explore step of the explore-then-exploit approach. This should not be confused with the interleaved explore and verify phases of the algorithm.
        """
        if self._exploring:
            # explore
            self._explorer.step()
            if self._explorer.is_finished():
                self._verifier = self._init_verifier()
                self._exploring = False
        else:
            # verify
            self._verifier.step()
            if self._verifier.is_finished():
                if not self._verifier.has_succeeded():
                    self._explorer.reset()
                    self._exploring = True
                    self.round += 1

    def exploration_finished(self) -> bool:
        """Determine whether the verification is completed."""
        return self._verifier.has_succeeded()

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the arm with the highest probability of being the first in a ranking of the arms.

        Returns
        -------
        Optional[int]
            The best arm, ``None`` if has not been calculated yet.
        """
        if self.exploration_finished():
            return self._explorer.get_best_arm()
        else:
            return None

    class CondorcetExplorer(Algorithm):
        """Implementation of an explorer for the Condorcet winner setting.

        Parameters
        ----------
        feedback_mechanism
            A ``FeedbackMechanism`` object describing the environment.
        failure_probability
            The probability of returning an incorrect result.
        random_state
            Used for random tie breaking.

        Attributes
        ----------
        failure_probability
        random_state
        pair_is_active
            Pairs of arms currently under investigation.
        preference_estimate
            Estimates for arm preferences.
        """

        def __init__(
            self,
            feedback_mechanism: FeedbackMechanism,
            failure_probability: float,
            random_state: Optional[np.random.RandomState],
        ) -> None:
            super().__init__(feedback_mechanism=feedback_mechanism, time_horizon=None)
            self.failure_probability = failure_probability
            self.random_state = (
                random_state if random_state is not None else np.random.RandomState()
            )
            num_arms = self.wrapped_feedback.get_num_arms()
            # mark all pairs without duplicates
            self.pair_is_active = np.ones((num_arms, num_arms)) - np.eye(num_arms) > 0
            # Note that the confidence radius is half of that in the paper since they consider the larger interval [-1, 1]
            self._confidence_radius = HoeffdingConfidenceRadius(
                failure_probability=self.failure_probability,
                probability_scaling_factor=lambda samples: 2
                * samples ** 2
                * num_arms ** 2,
            )
            self.preference_estimate = PreferenceEstimate(
                num_arms=num_arms, confidence_radius=self._confidence_radius
            )
            self._best_arm = None
            self._adversaries = None
            self._is_finished = False

        def reset(self) -> None:
            """Reset the exploration."""
            num_arms = self.wrapped_feedback.get_num_arms()
            self.pair_is_active = np.ones((num_arms, num_arms)) - np.eye(num_arms) > 0
            self.preference_estimate = PreferenceEstimate(
                num_arms=num_arms, confidence_radius=self._confidence_radius
            )
            self._best_arm = None
            self._adversaries = None
            self._is_finished = False

        def _query_pairs(self) -> None:
            """Duel every active pair of arms once.

            The order of the arms does not matter, even if both are present they should only be compared once.
            """
            num_arms = self.wrapped_feedback.get_num_arms()
            for arm1 in range(num_arms):
                for arm2 in range(arm1):
                    if (
                        self.pair_is_active[arm1, arm2]
                        | self.pair_is_active[arm2, arm1]
                    ):
                        result = self.wrapped_feedback.duel(arm1, arm2)
                        self.preference_estimate.enter_sample(arm1, arm2, result)

        def step(self) -> None:
            """Advance the algorithm by one step."""
            self._query_pairs()
            lower_bound = (
                2 * self.preference_estimate.get_lower_estimate_matrix().preferences - 1
            )
            upper_bound = (
                2 * self.preference_estimate.get_upper_estimate_matrix().preferences - 1
            )
            # update active arms
            # remove if lower bound is higher than 0
            # => AND with lower <= 0
            self.pair_is_active &= lower_bound <= 0
            # remove if upper bound is less than 0 and 2*upper<lower
            # AND with (upper >= 0 OR 2* upper  >= lower)
            self.pair_is_active &= np.logical_or(
                upper_bound >= 0, 2 * upper_bound >= lower_bound
            )
            # remove if lower bound is less than 0 and lower bound is greater than some other upper bound with the same first arm
            # AND (lower >= 0 OR lowerxy <= min(upperxy'))
            self.pair_is_active &= np.logical_or(
                lower_bound >= 0, lower_bound <= np.amin(upper_bound, axis=1)
            )

            # stop if no active arm pairs are left
            if np.sum(self.pair_is_active) == 0:
                self._is_finished = True
                candidates = np.argwhere(np.sum(lower_bound < 0, axis=1) == 0)
                self._best_arm = self.random_state.choice(
                    candidates.flatten()
                )  # there should only be one element here, but if the Condorcet winner does not exist there may be more
                best_arm_encoding = np.full(self.pair_is_active.shape[0], False)
                best_arm_encoding[self._best_arm] = True
                # Pick an adversary for each arm except the estimated Condorcet
                # winner. The adversary of arm x is that arm y against which it
                # has the lowest probability of winning in the upper-bound
                # estimation.
                corrected_upper_bound = upper_bound + np.eye(
                    self.wrapped_feedback.get_num_arms()
                )
                self._adversaries = np.ma.array(
                    np.argmin(corrected_upper_bound, axis=1), mask=best_arm_encoding
                )

        def is_finished(self) -> bool:
            """Determine whether the verification is completed."""
            return self._is_finished

        def get_best_arm(self) -> Optional[int]:
            """Return the best arm, if it has been determined.

            Returns
            -------
            Optional[int]
                The index of the best arm or ``None``, if not finished.
            """
            if self.is_finished():
                return self._best_arm
            else:
                return None

        def get_adversaries(self) -> Optional[List[int]]:
            """Return the adversary for each arm.

            Returns
            -------
            Optional[List[int]]
                A list containing the adversary indices assigned to the arms, ``None`` if not finished.
            """
            if self.is_finished():
                return self._adversaries
            else:
                return None

    class CondorcetVerifier(Algorithm):
        """Implementation of a verifier for the Condorcet winner setting.

        Parameters
        ----------
        feedback_mechanism
            A ``FeedbackMechanism`` object describing the environment.
        best_arm
            The best arm as computed by the exploration step.
        adversaries
            An adversary arm for each arm (the adversary of the ``best_arm`` is ignored).
        failure_probability
            The probability of returning an incorrect result.

        Attributes
        ----------
        best_arm
        adversaries
        failure_probability
        pair_is_active
            Pairs of arms currently under investigation.
        preference_estimate
            Estimates for arm preferences.
        """

        def __init__(
            self,
            feedback_mechanism: FeedbackMechanism,
            best_arm: int,
            adversaries: List[int],
            failure_probability: float,
        ) -> None:
            super().__init__(feedback_mechanism=feedback_mechanism, time_horizon=None)
            self.best_arm = best_arm
            self.adversaries = adversaries
            self.failure_probability = failure_probability
            num_arms = self.wrapped_feedback.get_num_arms()

            self.pair_is_active = np.eye(num_arms)[self.adversaries] > 0
            self.pair_is_active[self.best_arm, :] = False
            # Note that the confidence radius is half of that in the paper since they consider the larger interval [-1, 1]
            confidence_radius = HoeffdingConfidenceRadius(
                failure_probability=self.failure_probability,
                probability_scaling_factor=lambda samples: 2
                * samples ** 2
                * num_arms ** 2,
            )
            self.preference_estimate = PreferenceEstimate(
                num_arms=num_arms, confidence_radius=confidence_radius
            )
            self._is_finished = False
            self._success = False

        def query_pairs(self) -> None:
            """Duel every active pair of arms once.

            The order of the arms does not matter, even if both are present they should only be compared once.
            """
            num_arms = self.wrapped_feedback.get_num_arms()
            for arm1 in range(num_arms):
                for arm2 in range(arm1):
                    if (
                        self.pair_is_active[arm1, arm2]
                        or self.pair_is_active[arm2, arm1]
                    ):
                        self.preference_estimate.enter_sample(
                            arm1, arm2, self.wrapped_feedback.duel(arm1, arm2)
                        )

        def step(self) -> None:
            """Advance the algorithm by one step."""
            if self.is_finished():
                return
            self.query_pairs()
            lower_bound = (
                self.preference_estimate.get_lower_estimate_matrix().preferences
            )
            upper_bound = (
                self.preference_estimate.get_upper_estimate_matrix().preferences
            )
            # remove if upperxy(x) < 0.5
            # => AND with upper >= 0.5
            self.pair_is_active &= upper_bound >= 0.5
            # terminate with fail if lower bound against adversary is > 0.5
            if np.sum(lower_bound[self.pair_is_active] > 0.5, axis=0) > 0:
                self._is_finished = True
                self._success = False
            if np.sum(self.pair_is_active) == 0:
                self._is_finished = True
                self._success = True

        def is_finished(self) -> bool:
            """Determine whether the verification is completed."""
            return self._is_finished

        def has_succeeded(self) -> bool:
            """Determine if the verification succeeded, if it completed.

            Returns
            -------
            bool
                True for success.
            """
            return self._success
