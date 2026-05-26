"""A simple example of how a decision-tool using duelpy could work."""

from typing import Optional

from duelpy.algorithms.savage import Savage
from duelpy.feedback import CommandlineFeedback
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.util.feedback_decorators import FeedbackMechanismDecorator


class PreferencePrintingFeedback(FeedbackMechanismDecorator):
    """Prints a preference estimate before each duel.

    The ``preference_estimate`` attribute needs to be set after initialization.
    It is not passed in the constructor since it usually can only be accessed
    after the algorithm has been initialized.

    Parameters
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.

    Attributes
    ----------
    feedback_mechanism
        The ``FeedbackMechanism`` object to delegate to.
    preference_estimate
        The preference estimate that should be printed.
    """

    def __init__(self, feedback_mechanism: FeedbackMechanism) -> None:
        super().__init__(feedback_mechanism)
        self.preference_estimate: Optional[PreferenceEstimate] = None

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Print the preference estimate and then duel two arms.

        The actual duel is delegated to the underlying preference estimate.

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
        if self.preference_estimate is not None:
            print("Preference estimate is now")
            print(self.preference_estimate)
        return super().duel(arm_i_index, arm_j_index)


def _run_decision_experiment() -> None:
    arms = [
        "The best arm.",  # 0
        "The second best arm.",  # 1
        "The other second best arm, flip a coin.",  # 2
        "Third best arm.",  # 3
        "Least favorite arm.",  # 4
    ]
    pp_feedback = PreferencePrintingFeedback(CommandlineFeedback(arms))
    algorithm = Savage(feedback_mechanism=pp_feedback, failure_probability=0.5)
    # The preference estimate is initialized by the algorithm. Many algorithms
    # have this attribute, but it is not part of the `Algorithm` API. We rely
    # on the implementation of Savage here.
    pp_feedback.preference_estimate = algorithm.preference_estimate
    algorithm.run()
    print("Estimated Copeland winner:")
    print(algorithm.get_copeland_winner())


if __name__ == "__main__":
    _run_decision_experiment()
