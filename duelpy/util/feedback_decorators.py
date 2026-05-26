"""Decorators to alter the behavior of feedback mechanisms."""

from typing import Dict
from typing import List
from typing import Optional

from duelpy.feedback import FeedbackMechanism
from duelpy.stats.metrics import Metric
from duelpy.util.exceptions import AlgorithmFinishedException


class FeedbackMechanismDecorator(FeedbackMechanism):
    """A feedback mechanism that delegates to another feedback mechanism.

    This is intended  to be used as a base class for wrappers that want to
    "inject" some behavior or checks into an existing feedback mechanism. See
    :class:`BudgetedFeedbackMechanism<duelpy.util.feedback_decorator.BudgetedFeedbackMechanism>` for an example.

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.
    """

    def __init__(self, feedback_mechanism: FeedbackMechanism) -> None:
        # We override all functions and delegate to an existing feedback
        # mechanism. Therefore it does not make much sense to call the super
        # constructor here.
        # pylint: disable=super-init-not-called
        self.feedback_mechanism = feedback_mechanism

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        return self.feedback_mechanism.duel(arm_i_index, arm_j_index)

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
    ) -> int:
        """Perform multiple duels between two arms in a single step.

        This function allows for a more efficient implementation for repeating duels of the same arm pair.
        The fallback if this implementation is not changed are repeated calls of ``duel``.

        Parameters
        ----------
        arm_i_index
            The arm of challenger arm.
        arm_j_index
            The index of arm to compare against.
        duel_count
            The number of rounds ``arm_i_index`` is compared against ``arm_j_index``.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        return self.feedback_mechanism.duel_repeatedly(
            arm_i_index, arm_j_index, duel_count
        )

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.feedback_mechanism.get_arms()

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return self.feedback_mechanism.get_num_arms()


class MetricKeepingFeedbackMechanism(FeedbackMechanismDecorator):
    """A feedback mechanism that updates a set of metrics on every duel.

    This can be used if you want to keep track on some aspects of an algorithms
    performance during its execution.

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.
    metrics
        A dictionary of metrics to apply, keyed by their name.
    sample_interval
        The number of time steps per sample.

    Attributes
    ----------
    feedback_mechanism
    metrics
    sample_interval
    results
        A dictionary of lists, keyed by the names of the metrics.

    Examples
    --------
    Define a very simple preference-based multi-armed bandit problem through a
    preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> from duelpy.stats.preference_matrix import PreferenceMatrix
    >>> import numpy as np
    >>> random_state = np.random.RandomState(42)
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.8],
    ...     [0.2, 0.5],
    ... ]))
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)

    Now let's run an algorithm on this problem and keep track of the cumulative
    regret:

    >>> from duelpy.algorithms import Savage
    >>> from duelpy.stats.metrics import AverageRegret, Cumulative
    >>> metric_keeping_feedback = MetricKeepingFeedbackMechanism(
    ...     feedback_mechanism,
    ...     metrics={"average_regret": Cumulative(AverageRegret(preference_matrix))},
    ... )
    >>> algorithm = Savage(metric_keeping_feedback)
    >>> algorithm.run()
    >>> metric_keeping_feedback.results
    {'average_regret': [0.150..., 0.300...]}
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        metrics: Dict[str, Metric],
        sample_interval: int = 1,
    ):
        super().__init__(feedback_mechanism)
        self.duels_conducted = 0
        self.metrics = metrics
        self.sample_interval = sample_interval
        self.results: Dict[str, List[float]] = {key: [] for key in metrics.keys()}
        # Always add an implicit "time step metric". Can be seen as a key or
        # index for the other metrics.
        self.results["time_step"] = []
        # self.filename = "data.txt"
        # with open(self.filename, 'a') as file:
        #     file.write("history:\n")

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Raises
        ------
        AlgorithmFinishedException
            If the budget would be exceeded by this duel.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        self.duels_conducted += 1
        result = super().duel(arm_i_index, arm_j_index)
        # with open(self.filename, 'a') as file:
        #     file.write(str(arm_i_index) + "," + str(arm_j_index)+"\n")
        #print(arm_i_index + "," + arm_j_index+ "\n")
        for (name, metric) in self.metrics.items():
            # Always call the metric, in case it keeps some internal state.
            metric_value = metric(arm_i_index, arm_j_index)
            if self.duels_conducted % self.sample_interval == 0:
                self.results[name].append(metric_value)
        if self.duels_conducted % self.sample_interval == 0:
            self.results["time_step"].append(self.duels_conducted)
        return result


class BudgetedFeedbackMechanism(FeedbackMechanismDecorator):
    """A feedback mechanism wrapper that ensures a duel budget is not exceeded.

    This can be used to provide an upper-bound on the number of duels that some
    function (that may be out of the algorithm's control) can perform. Examples
    are calls to sorting algorithms or other multi-armed bandit algorithms.
    Using this wrapper is different from directly passing a ``time_horizon``
    because this only provides a lower, but not an upper bound on the number
    duels.

    Examples
    --------
    Define a very simple preference-based multi-armed bandit problem through a
    preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> import numpy as np
    >>> random_state = np.random.RandomState(42)
    >>> preference_matrix = np.array([
    ...     [0.5, 0.7],
    ...     [0.3, 0.5],
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)

    Now run a PAC algorithm on it in various different configurations. First
    let's try to run it unmodified and without a time horizon:

    >>> from duelpy.algorithms import Savage
    >>> pac_algorithm = Savage(feedback_mechanism)
    >>> pac_algorithm.run()
    >>> pac_algorithm.wrapped_feedback.duels_conducted
    2
    >>> pac_algorithm.get_copeland_winner()
    0

    Now let's say we only have one more duel to spare. We can use the wrapper
    for that:

    >>> budgeted_feedback = BudgetedFeedbackMechanism(feedback_mechanism, max_duels=1)
    >>> pac_algorithm = Savage(budgeted_feedback)
    >>> try:
    ...     pac_algorithm.run()
    ... except budgeted_feedback.exception_class:
    ...     # The algorithm was not able to find a Copeland winner with the limited duel budget.
    ...     pass
    >>> pac_algorithm.wrapped_feedback.duels_conducted # Just one additional duel
    1
    >>> pac_algorithm.get_copeland_winner() is None  # Algorithm terminated early
    True

    But if the algorithm is able to complete within the budget, the behavior is
    unchanged:

    >>> budgeted_feedback = BudgetedFeedbackMechanism(feedback_mechanism, max_duels=100)
    >>> pac_algorithm = Savage(budgeted_feedback)
    >>> try:
    ...     pac_algorithm.run()
    ... except budgeted_feedback.exception_class:
    ...     # This should not happen, the budget is sufficiently large
    ...     assert False
    >>> pac_algorithm.wrapped_feedback.duels_conducted
    2
    >>> pac_algorithm.get_copeland_winner()
    0

    Which is different from how the algorithm would behave if we would pass a
    time horizon instead:

    >>> pac_algorithm = Savage(feedback_mechanism, time_horizon=100)
    >>> pac_algorithm.run()
    >>> # Time horizon is both an upper and a lower bound.
    >>> pac_algorithm.wrapped_feedback.duels_conducted
    100
    >>> pac_algorithm.get_copeland_winner()
    0

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.
    max_duels
        The number of duels that are allowed to be conducted through this
        decorator. May be ``None``, in which case no budget is applied.
    """

    def __init__(
        self, feedback_mechanism: FeedbackMechanism, max_duels: Optional[int]
    ) -> None:
        super().__init__(feedback_mechanism)
        self.max_duels = max_duels
        self.duels_conducted = 0

        # This makes it possible to only catch the AlgorithmFinishedExceptions
        # thrown by this object. This can be useful when multiple wrappers are
        # nested (e.g. one algorithm uses another, both use
        # BudgetedFeedbackMechanism) and the exception should be caught by the
        # algorithm that set the limit.
        class TimeBudgetExceededException(AlgorithmFinishedException):
            """Raised if the duel budget would be exceeded by the current duel."""

        self.exception_class = TimeBudgetExceededException

    def duels_exhausted(self) -> bool:
        """Determine if the duel budget has been reached.

        Returns
        -------
        bool
            True if the number of duels that have been conducted through this
            decorator has reached the given budget.
        """
        return self.max_duels is not None and self.duels_conducted >= self.max_duels

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Raises
        ------
        TimeBudgetExceededException
            If the budget would be exceeded by this duel. The exception class
            is local to the object. Different instances raise different
            exceptions. The exception class of an instance can be accessed
            through the ``exception_class`` attribute.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        if self.duels_exhausted():
            raise self.exception_class()
        result = super().duel(arm_i_index, arm_j_index)
        self.duels_conducted += 1
        return result

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
    ) -> int:
        """Perform multiple duels between two arms in a single step.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.
        duel_count
            Number of duels that has to be performed.

        Raises
        ------
        TimeBudgetExceededException
            If the budget would be exceeded by this duel. The exception class
            is local to the object. Different instances raise different
            exceptions. The exception class of an instance can be accessed
            through the ``exception_class`` attribute.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        if (
            self.max_duels is not None
            and self.duels_conducted + duel_count >= self.max_duels
        ):
            duel_count = max(0, self.max_duels - self.duels_conducted)
        self.duels_conducted += duel_count

        wins = super().duel_repeatedly(arm_i_index, arm_j_index, duel_count)

        if self.duels_exhausted():
            raise self.exception_class()

        return wins
