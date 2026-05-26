"""Gather feedback from a human on the commandline."""

from duelpy.feedback.feedback_mechanism import FeedbackMechanism


class CommandlineFeedback(FeedbackMechanism):
    """Compare two arms based on human feedback on the CLI."""

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms based on human feedback.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        print(
            f'Do you prefer arm "{self.arms[arm_i_index]}" (i) or arm "{self.arms[arm_j_index]}" (j)?'
        )
        result = input("[i/j] ")
        while result not in {"i", "j"}:
            print('Please choose one of "i" or "j"')
            result = input("[i/j] ")
        arm_i_wins = result == "i"
        return arm_i_wins
