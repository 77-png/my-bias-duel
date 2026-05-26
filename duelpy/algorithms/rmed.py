"""An implementation of the Relative Minimum Empirical Divergence for Dueling Bandits."""
from copy import deepcopy
from itertools import combinations
from typing import Callable
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from scipy.special import rel_entr

from duelpy.algorithms.algorithm import Algorithm
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.util.utility_functions import argmin_set


class Rmed1(Algorithm):
    r"""Implementation of Relative Minimum Empirical Divergence 1 algorithm.

    The goal of this algorithm is to minimize regret through the relative comparison of the arms.

    This algorithm assumes there exists a :term:`Condorcet winner`.

    The expected regret is upper bounded by :math:`O(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},1/2)})+O(K^{2+\epsilon})`
    where :math:`KL(p, q)` is the Kullback-Leibler divergence of Bernoulli
    random variables with parameters :math:`p` and :math:`q`, :math:`\epsilon>0` is exploratory constant, a
    parameter of the algorithm and :math:`q_{i^*,i}` is prefrence probability of the best arm (:math:`i^*`) compared to any arm
    (:math:`i`). The asymtotically optimal lower regret bound K-duel bandit problem is
    :math:`\Omega(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},1/2)})`.
    Rmed algorithms, presented in paper :cite:`komiyama15regret`, is based on empirical divergance where empirical
    divergance of arm :math:`a_i` at time t is :math:`I_{a_i}(t)=\sum_{a_j:q_{i,j}^t \leq 1/2} n_{i,j}^t KL(q_{i,
    j}^t,1/2)` , :math:`n_{i,j}` number of comparision of :math:`a_i` and :math:`a_j` , :math:`KL(q_{i,j}^t,
    1/2)` is kullback-Leibler divergence of Bernoulli random variables with parameters :math:`p` and :math:`q`,
    where :math:`p` is the probability of one arm being better than the other and :math:`q` is the probability of
    drawn arm that is likely to be the :term:`Condorcet winner` with high probability. Initially, unique pairs of
    arms are generated and compared for exploration. Each unique pair of arms is compared for :math:`L` times,
    where :math:`L=1` for Rmed1. At the end of the initial exploration phase, the timestep is equal to
    :math:`t=\frac{L(K-1)K}{2}` where K is number of arms. At each subsequent step, the set of remaining arms is
    reduced to get a set of likely Condorcet winners or a :term:`Condorcet winner`.

    The second phase proceeds by creating a set of arms as follows: ``arm_current_loop_set`` (:math:`L_C`) tracks the
    arm in the current loop in each step, ``arm_remaining_loop_set`` (:math:`L_R`) the arms remaining after the loop
    and ``arm_condorcet_candidate_set`` (:math:`L_N`) contains the Condercet candidate arms in each loop. In each
    step, the loop continues until all the arms have been selected from :math:`L_C`. Initially, all the arms are
    present in :math:`L_C` and :math:`L_R`. Meanwhile, :math:`L_N`  is empty. At each step, :math:`L_C` contains the
    remaining set of arms. In each iteration, a likely winner arm is selected from :math:`L_C`. After  that a
    challenger arm is to be selected for the likely winner. For the selection of a challenger arm, an arm is selected
    from the ``opponent set``. The ``opponent set`` is the set of arms for which the estimated preference probability
    compared to the reference arm arm is greater than :math:`0.5`. If the opponent set is empty, the ``best arm`` is
    selected as the challenger arm else an arm with minimum preference estimate is selected. The ``best arm`` is the
    arm which has the minimum empirical divergence among all the arms at the current timestep. Next the candidate and
    the challenger are dueled. The candidate is removed from the :math:`L_R`. The likely winner arm is verified
    whether the arm is the potential :term:`Condorcet winner`. If it is valid, it is added to the list of potential
    :term:`Condorcet winner` set :math:`L_N` and the loop continues. After the end of the loop the active arms (
    :math:`L_C` and :math:`L_R`) are set to the Condorcet candidates of the current loop (:math:`L_N`) while the
    Condorcet candidates are reset for the next iteration. winner` set :math:`L_N` and the loop continues. After the
    end of the loop the active arms (:math:`L_C` and :math:`L_R`) are set to the Condorcet candidates of the current
    loop (:math:`L_N`) while the Condorcet candidates are reset for the next iteration.  If the time horizon is not
    yet reached the algorithm takes the next step with the updated values of :math:`L_C`, :math:`L_R` and :math:`L_N`.

    Based on some studies, the K-armed dueling bandit problem has a lower regret bound of :math:`\Omega(K \log T)`.
    In the paper :cite:`komiyama15regret`,The algorithm has been shown to reach this lower bound math:`\Omega(K \log
    T)` under the Condorcet assumption, even up to the constant factor. Also, RMED is the first asymptotically
    optimal algorithm in the study of the dueling bandit problem.


    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    random_state
        Optional, used for random choices in the algorithm.
    divergence_tolerance
        Optional, Any non-negative function :math:`f(K)` that is independent of timestep, where :math:`K` is the number
        of arms. It is  is used to determine the size of the allowed "empirical divergence" gap between the currently
        estimated best arm and the remaining candidate arms . The "tolerance" or "empirical divergence" gap does not
        only depend on this function, but also on the current time step. For more details refer to equation 4
        in section 3.1 of :cite:`komiyama15regret`. Default value is taken from the experiment which is
        :math:`f(K)=0.3*K^(1.01)`.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> from duelpy.stats.metrics import AverageRegret
    >>> from duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"average_regret": AverageRegret(preference_matrix)}
    ... )
    >>> divergence_tolerance = lambda num_arms: 0.3 * np.power(num_arms, 1.01)
    >>> test_object = Rmed1(
    ...     feedback_mechanism,
    ...     divergence_tolerance=divergence_tolerance,
    ...     random_state=random_state,
    ...     time_horizon=100
    ... )
    >>> test_object.run()
    >>> np.round(np.sum(feedback_mechanism.results["average_regret"]), 2)
    20.8
    """

    # pylint: disable=too-many-instance-attributes
    # the attribute helps readability here
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        divergence_tolerance: Optional[Callable[[int], float]] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(
            feedback_mechanism=feedback_mechanism, time_horizon=time_horizon
        )

        self.divergence_tolerance = (
            divergence_tolerance
            if divergence_tolerance is not None
            else lambda num_arms: 0.3 * np.power(num_arms, 1.01)
        )

        self.preference_estimate = PreferenceEstimate(
            num_arms=self.wrapped_feedback.get_num_arms()
        )
        self.time_step = 0
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        # draw each pair of arms L times. L is denoted as self.num_initial_comparisons
        self.num_initial_comparisons = 1

        arms = self.wrapped_feedback.get_arms()
        # arms in the the current loop is arbitrarily fixed order.
        self.random_state.shuffle(arms)
        self.arms_current_loop = deepcopy(arms)
        # arms remaining after a current loop
        self.arms_remaining_current_loop = deepcopy(arms)
        self.arms_condorcet_candidate_current_loop: List[int] = list()
        # combination of the pair of the arms
        self.unique_arm_pairs = list(combinations(arms, 2))
        # assume the best arm to be one out of list of the arms which has minimum empirical divergence
        self.best_arm = 0
        self.empirical_divergence = np.zeros(self.wrapped_feedback.get_num_arms())
        # required for Rmed2FH
        self.fix_challenger = [0] * self.wrapped_feedback.get_num_arms()
        self.opponent_set: Set = set()

    def _initial_phase_comparison(self) -> None:
        """Run initial phase of the comparison between the arm pairs provided number of loops."""
        for arm_i, arm_j in self.unique_arm_pairs:
            for _ in range(self.num_initial_comparisons):
                self.time_step += 1
                self.preference_estimate.enter_sample(
                    arm_i, arm_j, self.wrapped_feedback.duel(arm_i, arm_j)
                )

    def _get_candidate_target(self, arm_i: int) -> int:
        """Select the challenger arm for reference arm.

        The list of the challenger arm is also called as opponent set. The opponent set is set of challenger arm whose
        preference estimate of selected reference  arm with challenger arm is 0.5. If the opponent set length is 0
        or the current best arm exist in opponent set then the current best arm is only the challenger arm. Meanwhile,
        if above condition does not matches than the challenger arm is selected from the opponent set which has
        minimum preference estimate. Ties are broken randomly.

        Parameters
        ----------
        arm_i
            The reference arm (first candidate).

        Returns
        -------
        int
            The challenger arm (second candidate) for reference arm (first candidate).
        """
        arm_i_vs_arm_j_mean_estimates = (
            self.preference_estimate.get_mean_estimate_matrix().preferences[arm_i][:]
        )
        self.opponent_set = set(
            np.argwhere(arm_i_vs_arm_j_mean_estimates <= 0.5)[0]
        ) - {arm_i}

        self.best_arm = self._get_best_arm()

        if len(self.opponent_set) == 0 or self.best_arm in self.opponent_set:
            return self.best_arm

        return self.random_state.choice(
            argmin_set(arm_i_vs_arm_j_mean_estimates, [arm_i])
        )

    def _get_best_arm(self) -> int:
        """Calculate the arm which has minimum empirical divergence.

        Returns
        -------
        int
            The arm with minimum empirical divergence.
        """
        for arm_i in self.wrapped_feedback.get_arms():
            self.empirical_divergence[arm_i] = self._get_arm_empirical_divergence(arm_i)

        return self.random_state.choice(argmin_set(self.empirical_divergence))

    def _get_arm_empirical_divergence(self, arm_i: int) -> float:
        """Calculate the empirical divergence of an arm.

        Parameters
        ----------
        arm_i
            The arm whose empirical divergence to be calculated.

        Returns
        -------
        float
           The empirical divergence.
        """
        empirical_divergence_arm_i = 0
        for arm_j in self.opponent_set:
            mean_estimate = self.preference_estimate.get_mean_estimate(arm_i, arm_j)
            kl_divergences = (
                rel_entr(
                    1 - mean_estimate,
                    0.5,
                )
                + rel_entr(mean_estimate, 0.5)
            )
            empirical_divergence_arm_i += (
                self.preference_estimate.get_num_samples(arm_i, arm_j) * kl_divergences
            )
        return empirical_divergence_arm_i

    def _is_condorcet_winner_candidate(self, arm_i: int) -> bool:
        """Check whether the arm is the candidate for the Condorcet winner.

        This corresponds to equation 4 in section 3.1 of :cite:`komiyama15regret`.

        Parameters
        ----------
        arm_i
            An arm.

        Returns
        -------
        bool
           True if the arm is candidate for the Condorcet winner.
        """
        return (
            self.empirical_divergence[arm_i] - self.empirical_divergence[self.best_arm]
        ) <= (
            np.log(self.time_step)
            + self.divergence_tolerance(self.wrapped_feedback.get_num_arms())
        )

    def step(self) -> None:
        """Each step of the algorithm after the some initial drawn pairs of arm for the competition."""
        if self.time_step == 0:
            # duels on first call of step function
            self._initial_phase_comparison()
            self._fix_candidate_target_arm()
        # initial number of duels on each call of step function
        self._each_step_initial_comparison()
        # check whether time step have reach to time horizon

        for arm_i in self.arms_current_loop:
            self.time_step += 1
            arm_j = self._get_candidate_target(arm_i)
            self.preference_estimate.enter_sample(
                arm_i, arm_j, self.wrapped_feedback.duel(arm_i, arm_j)
            )
            self.arms_remaining_current_loop.remove(arm_i)

            for remaining_arm in self.wrapped_feedback.get_arms():
                if (
                    remaining_arm not in self.arms_remaining_current_loop
                    and remaining_arm not in self.arms_condorcet_candidate_current_loop
                    and self._is_condorcet_winner_candidate(remaining_arm)
                ):
                    self.arms_condorcet_candidate_current_loop.append(remaining_arm)

        self.arms_current_loop = deepcopy(self.arms_condorcet_candidate_current_loop)
        self.arms_remaining_current_loop = deepcopy(
            self.arms_condorcet_candidate_current_loop
        )
        self.arms_condorcet_candidate_current_loop = list()

    def _each_step_initial_comparison(self) -> None:
        # Do not perform the initial comparisons that RMED2 would perform.
        pass

    def _fix_candidate_target_arm(self) -> None:
        # Do not fix the target to the reference arm that RMED2FH would do.
        pass


def _kullback_leibler_divergence_plus(
    probability_p: float, probability_q: float
) -> float:
    r"""Compute KL divergence plus for Bernoulli distributions.

    The KL-divergence plus (for two Bernoulli random variables with parameters :math:`p` and :math:`q`)
    from :math:`q` to :math:`p` is formulated as :math:`d^+(p,q) = d(p,q)` if :math:`p<q` else ``0``.
    Here :math:`d` is the regular KL-divergence.

    Parameters
    ----------
    probability_p
        The preference probability of one arm over another.
    probability_q
        The preference probability of one arm over another.

    Returns
    -------
    float
        The KL-Divergence plus of the two Bernoulli distributions.
    """
    if probability_p < probability_q:
        return rel_entr(1 - probability_p, probability_q,) + rel_entr(
            probability_p,
            probability_q,
        )
    return 0


class Rmed2(Rmed1):
    r"""Implementation of Relative Minimum Empirical Divergence 2 algorithm.

    The goal of this algorithm is to minimize regret through the relative comparison of the arms.

    This algorithm assumes there exists a :term:`Condorcet winner`. The preference matrix is estimated through a series of
    relative comparison.

    The expected regret is upper bounded by :math:`O(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},1/2)}) + O(K^{2+\epsilon})`
    where :math:`KL(p,q)` is the Kullback-Leibler divergence of Bernoulli random variables with
    parameters :math:`p` and :math:`q`, :math:`\epsilon>0` is exploratory constant, a parameter of the algorithm and
    :math:`q_{i^*,i}` is prefrence probability of best arm(:math:`i^*`) with any arm (:math:`i`). The asymtotically
    optimal lower regret bound K-duel bandit problem is :math:`\Omega(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},
    1/2)})`.

    The working of the algorithm Rmed2, presented in paper :cite:`komiyama15regret`, is similar to the Rmed1
    algorithm. It slightly differs in explore phase where Initially each unique pair of arm is dueled :math:`L`
    times, where :math:`L=\lceil \alpha \log{\log{T}} \rceil`, and T is the time horizon. In each algorithm step,
    there is explore and exploit phase. In explore phase, all the unique pairs are drawn again until the number of
    duels (N) is less than or equal to :math:`\alpha \log{ \log{t}}` where :math:`t` is the current time step. The
    exploit phase is similar to the Rmed1, the only difference is selecting the challenger arm. The concept of
    divergence plus is introduced. Select challenger arm from the arm pool which has the minimum divergence plus with
    selected likely winner arm. If the challenger arm meets the condition to be challenger arm, it is a candidate
    target else  the challenger arm is selected  as that of Rmed1 algorithm.

    As to match the asymptotical regret lower bound, Rmed1 is modified to Rmed2 algorithm with a constant factor.
    Rmed2 uses the same subroutine as Rmed1, but it only differs on how the comparison target is selected to the
    first chosen arm. Rmed2 tries to select the arm that is most likely to be the Condorcet winner in most of the
    algorithm's rounds/step . It also explores in order to reduce regret, but sometimes the regret increases if it
    fails to estimate the real :term:`Condorcet winner`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    exploratory_constant
        Optional, The exploratory constant is related to the number of times each arm pair is compared. Corresponds to
        :math:`\alpha` in paper. The value of ``exploratory_constant`` must be greater than :math:`0.5`. Default
        value is ``3``.
    random_state
        Optional, used for random choices in the algorithm.
    divergence_tolerance
        Optional, Any non-negative function :math:`f(K)` that is independent of timestep, where :math:`K` is the number
        of arms. It is  is used to determine the size of the allowed "empirical divergence" gap between the currently
        estimated best arm and the remaining candidate arms. The "tolerance" or "empirical divergence" gap does not
        only depend on this function, but also on the current time step. For more details refer to the equation
        section 3.1 in paper :cite:`komiyama15regret`. Default value is taken from the experiment which is
        :math:`f(K)=0.3*K^(1.01)`.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> from duelpy.stats.metrics import AverageRegret
    >>> from duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"average_regret": AverageRegret(preference_matrix)}
    ... )
    >>> divergence_tolerance = lambda num_arms: 0.3 * np.power(num_arms, 1.01)
    >>> test_object = Rmed2(
    ...     feedback_mechanism,
    ...     divergence_tolerance=divergence_tolerance,
    ...     random_state=random_state,
    ...     time_horizon=100
    ... )
    >>> test_object.run()
    >>> np.round(np.sum(feedback_mechanism.results["average_regret"]), 2)
    21.4
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        divergence_tolerance: Optional[Callable[[int], float]] = None,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 3,
    ):
        super().__init__(
            divergence_tolerance=divergence_tolerance,
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
        )
        self.exploratory_constant = exploratory_constant
        self.num_initial_comparisons = int(
            max(np.ceil(exploratory_constant * np.log(np.log(self.time_horizon))), 0)
        )

    def _each_step_initial_comparison(self) -> None:
        """Draw the arm pairs and compare them in each algorithm step.

        The pair of arms are drawn until the number of sample is less or equal to the product of exploratory
        constant and loglog(self.time).
        """
        for arm_i, arm_j in self.unique_arm_pairs:
            while self.preference_estimate.get_num_samples(
                arm_i, arm_j
            ) < self.exploratory_constant * np.log(np.log(self.time_step)):
                self.time_step += 1
                self.preference_estimate.enter_sample(
                    arm_i, arm_j, self.wrapped_feedback.duel(arm_i, arm_j)
                )

    def _get_candidate_target(self, arm_i: int) -> int:
        """Select the challenger arm for reference arm.

        Divergence plus is introduced. Select challenger arm from the arm pool which has minimum divergence plus with
        reference arm. if the challenger arm meet the condition to be candidate target it is a candidate target else
        the candidate target is calculated as in Rmed1.

        Parameters
        ----------
        arm_i
            The reference arm (first candidate).

        Returns
        -------
        int
            The challenger arm (second candidate) for reference arm (first candidate).
        """
        challenger_arm = self._get_challenger_arm(arm_i)
        # If the mean estimate of the reference arm with challenger arm is greater than 0.5 then the challenger arm is
        # assumed to be in the opponent set.
        mean_estimate_arm_i_with_challenger = (
            self.preference_estimate.get_mean_estimate(arm_i, challenger_arm)
        )
        if (
            mean_estimate_arm_i_with_challenger > 0.5
            and self.preference_estimate.get_num_samples(arm_i, self.best_arm)
            >= (
                self.preference_estimate.get_num_samples(arm_i, challenger_arm)
                / np.log(np.log(self.time_step))
            )
        ):
            return challenger_arm

        return super()._get_candidate_target(arm_i)

    def _get_challenger_arm(self, arm_i: int) -> int:
        """Compute the challenger arm for reference arm.

        Parameters
        ----------
        arm_i
            The reference arm.

        Returns
        -------
        int
           The challenger arm for the reference arm.
        """
        empirical_plus_divergence = [
            self.empirical_divergence_plus(arm_i, arm_j)
            for arm_j in self.wrapped_feedback.get_arms()
        ]

        return self.random_state.choice(argmin_set(empirical_plus_divergence, [arm_i]))

    def empirical_divergence_plus(self, arm_i: int, arm_j: int) -> float:
        """Calculate the empirical divergence plus of an arm with reference arm.

        Parameters
        ----------
        arm_i
            The reference arm.
        arm_j
            The arm whose empirical divergence is to be calculated.

        Returns
        -------
        float
           The empirical divergence plus.
        """
        divergence_plus = _kullback_leibler_divergence_plus(
            self.preference_estimate.get_mean_estimate(arm_i, arm_j), 0.5
        )
        if divergence_plus == float("inf"):
            return 0
        elif divergence_plus == 0:
            return 1e308
        else:
            return (
                1
                - self.preference_estimate.get_mean_estimate(self.best_arm, arm_i)
                - self.preference_estimate.get_mean_estimate(self.best_arm, arm_j)
            ) / divergence_plus


class Rmed2FH(Rmed2):
    r"""Implementation of Relative Minimum Empirical Divergence 2 Fixed Horizon algorithm.

    The goal of this algorithm is to minimize regret through the relative comparison of the arms.

    This algorithm assumes there exists a :term:`Condorcet winner`. The preference matrix is estimated through a series
    of relative comparison.

    The expected regret is upper bounded by :math:`O(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},1/2)})+O(K^{2+\epsilon})`
    where :math:`KL(p,q)`  is the Kullback-Leibler divergence of Bernoulli
    random variables with parameters :math:`p` and :math:`q`, :math:`\epsilon>0` is exploratory constant,
    a parameter of the algorithm and :math:`q_{i^*,i}` is prefrence probability of best arm(:math:`i^*`) with any arm
    (:math:`i`). The asymtotically optimal lower regret bound K-duel bandit problem is
    :math:`\Omega(\sum_{i\neq i^*}\frac{\log T}{KL(q_{i^*,i},1/2)})`.

    It is the static version of RMED2 called RMED2 Fixed horizon (presented in paper :cite:`komiyama15regret` ).
    RMED2FH shows that it is an asymptotically optimal algorithm under Condorcet assumption. The working of the
    algorithm is similar to the Rmed2 algorithm. It slightly differs in explore phase where initially, each unique
    pair of arms is dueled :math:`L` time where :math:`L=1` like that in Rmed1. And, In each algorithm step,
    there is only an exploit phase. The exploit phase is similar to the Rmed1 only the difference is selecting the
    challenger arm. The concept of fixing challenger arm for each likely winner arm arises. For each likely winner
    arm , a challenger arm is selected from the arm pool which has the minimum divergence plus with likely winner
    arm. The challenger is fixed after the initial explore phase. If the challenger arm meet the condition to be
    challenger arm it is a candidate target else the challenger arm is selected as that of Rmed1 algorithm.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    exploratory_constant
        Optional, The exploratory constant is related to the number of times each arm pair is compared. Corresponds to
        :math:`\alpha` in paper. The value of ``exploratory_constant`` must be greater than :math:`0.5`. Default
        value is ``3``.
    random_state
        Optional, used for random choices in the algorithm.
    divergence_tolerance
        Optional, Any non-negative function :math:`f(K)` that is independent of timestep, where :math:`K` is the number
        of arms. It is  is used to determine the size of the allowed "empirical divergence" gap between the currently
        estimated best arm and the remaining candidate arms . The "tolerance" or "empirical divergence" gap does not
        only depend on this function, but also on the current time step. For more details refer to the equation
        section 3.1 in paper :cite:`komiyama15regret`. Default value is taken from the experiment which is
        :math:`f(K)=0.3*K^(1.01)`.

    Attributes
    ----------
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    exploratory_constant
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> from duelpy.stats.metrics import AverageRegret
    >>> from duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MetricKeepingFeedbackMechanism(
    ...     MatrixFeedback(preference_matrix, random_state=random_state),
    ...     metrics={"average_regret": AverageRegret(preference_matrix)}
    ... )
    >>> divergence_tolerance= lambda num_arms: 0.3 * np.power(num_arms, 1.01)
    >>> test_object = Rmed2FH(
    ...     feedback_mechanism,
    ...     divergence_tolerance=divergence_tolerance,
    ...     random_state=random_state,
    ...     time_horizon=100
    ... )
    >>> test_object.run()
    >>> np.round(np.sum(feedback_mechanism.results["average_regret"]), 2)
    21.8
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        divergence_tolerance: Optional[Callable[[int], float]] = None,
        random_state: Optional[np.random.RandomState] = None,
        exploratory_constant: float = 3,
    ):
        super().__init__(
            divergence_tolerance=divergence_tolerance,
            feedback_mechanism=feedback_mechanism,
            time_horizon=time_horizon,
            random_state=random_state,
            exploratory_constant=exploratory_constant,
        )

    def _each_step_initial_comparison(self) -> None:
        # Do not perform the initial comparisons that RMED2 would perform.
        pass

    def _fix_candidate_target_arm(self) -> None:
        """Fix the challenger arm for all reference arm during initialization of algorithm."""
        for arm_i in self.wrapped_feedback.get_arms():
            self.fix_challenger[arm_i] = self._get_challenger_arm(arm_i)

    def _get_candidate_target(self, arm_i: int) -> int:
        """Select the challenger arm for reference arm.

        Parameters
        ----------
        arm_i
            The reference arm (first candidate).

        Returns
        -------
        int
            The challenger arm (second candidate) for reference arm (first candidate).
        """
        challenger_arm = self.fix_challenger[arm_i]
        # If the mean estimate of the condorcet arm with challenger arm is greater than 0.5 then the challenger arm is
        # assumed to be in the opponent set.
        mean_estimate_arm_i_with_challenger = (
            self.preference_estimate.get_mean_estimate(arm_i, challenger_arm)
        )
        if (
            mean_estimate_arm_i_with_challenger > 0.5
            and self.preference_estimate.get_num_samples(arm_i, self.best_arm)
            >= self.preference_estimate.get_num_samples(arm_i, challenger_arm)
            / np.log(np.log(self.time_horizon))
        ):
            return self.fix_challenger[arm_i]
        # pylint: disable=protected-access
        # Rmed2FH is inherited from Rmed2, Rmed2 is inherited from Rmed1, So, super cannot be directly used.
        return Rmed1._get_candidate_target(self, arm_i)
