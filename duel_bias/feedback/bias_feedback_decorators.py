"""Decorators to alter the behavior of the feedback from biased evaluators.
"""

from typing import Dict, Union
from typing import List
from typing import Optional

from duelpy.feedback import FeedbackMechanism
from duelpy.stats.metrics import Metric
from duelpy.util.exceptions import AlgorithmFinishedException

# Biased feedback mechanisms
from duel_bias.feedback import MatrixFeedbackBias

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

    def __init__(self, feedback_mechanism: Union[FeedbackMechanism, MatrixFeedbackBias]) -> None:
        # We override all functions and delegate to an existing feedback
        # mechanism. Therefore, it does not make much sense to call the super
        # constructor here.
        # pylint: disable=super-init-not-called
        self.feedback_mechanism = feedback_mechanism

        # is_bias is the Indicator of whether the input feedback is an instance of biased feedback_mechanism
        if isinstance(feedback_mechanism, FeedbackMechanismDecorator):
            if hasattr(feedback_mechanism, 'is_bias'):
                self.is_bias = feedback_mechanism.is_bias
            else:
                # self.is_bias = False
                raise ValueError('An unexpected issue has arisen:'
                                    'If feedback_mechanism is an instance of FeedbackMechanismDecorator,'
                                    'then it should have the attribute of is_bias')
        else:
            if isinstance(feedback_mechanism, MatrixFeedbackBias):
                self.is_bias = True
            else:
                self.is_bias = False

    def duel(self, arm_i_index: int, arm_j_index: int, user_k_index: int = None) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j

        Returns
        -------
        bool
            True if the first arm wins.
        """
        if self.is_bias:
            return self.feedback_mechanism.duel(arm_i_index, arm_j_index, user_k_index)
        else:
            return self.feedback_mechanism.duel(arm_i_index, arm_j_index)

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
        user_k_index: int = None,
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
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        if self.is_bias:
            return self.feedback_mechanism.duel_repeatedly(
                arm_i_index, arm_j_index, duel_count, user_k_index
            )
        else:
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

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.
    metrics
        A dictionary of metrics to apply, keyed by their name.
    sample_interval
        The number of time steps per sample.
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
        self.results: Dict[str, List[Union[float, None]]] = {key: [] for key in metrics.keys()}
        self.results["time_step"] = []
    
    def duels_exhausted(self):
        return self.feedback_mechanism.duels_exhausted()

    def duel(self, arm_i_index: int, arm_j_index: int, user_k_index: int = None) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j

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
        result = super().duel(arm_i_index, arm_j_index, user_k_index)
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

    def duel(self, arm_i_index: int, arm_j_index: int, user_k_index: int = None) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j

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
        result = super().duel(arm_i_index, arm_j_index, user_k_index)
        self.duels_conducted += 1
        return result

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
        user_k_index: int = None,
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
        user_k_index
            (Optional) The user to judge the winner between arm_i and arm_j

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

        wins = super().duel_repeatedly(arm_i_index, arm_j_index, duel_count, user_k_index)

        if self.duels_exhausted():
            raise self.exception_class()

        return wins
