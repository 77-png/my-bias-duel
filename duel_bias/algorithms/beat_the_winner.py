"""Implementation of the 'Beat the Winner' and 'Modified Beat the Winner' algorithms.
"""

from typing import Optional
import numpy as np
from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.feedback import FeedbackMechanism

class BeatTheWinner(CondorcetProducer):
    """Implementation of the 'Beat the Winner' algorithm
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: np.random.default_rng = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.default_rng(42)
        )
        self.num_arms = self.wrapped_feedback.get_num_arms()
        self.arm_sequence = list(self.random_state.permutation(self.num_arms))
        self.rounds = 1
        self.i_wins = 0
        self.j_wins = 0

    def _reset_match(self):
        self.rounds += 1
        self.i_wins = 0
        self.j_wins = 0

    def step(self) -> None:
        arm_i, arm_j = self.arm_sequence[:2]

        if self.wrapped_feedback.duel(arm_i, arm_j):
            self.i_wins += 1
        else:
            self.j_wins += 1

        if self.i_wins >= self.rounds:
            self._reset_match()
            self.arm_sequence = [arm_i] + self.arm_sequence[2:] + [arm_j]
        elif self.j_wins >= self.rounds:
            self._reset_match()
            self.arm_sequence = [arm_j] + self.arm_sequence[2:] + [arm_i]

    def get_condorcet_winner(self) -> int:
        """Get the index of the arm currently believed to be the Condorcet winner.
        """
        return self.arm_sequence[0]

class ModifiedBeatTheWinner(CondorcetProducer):
    """Implementation of the 'Modified Beat the Winner' algorithm
    """
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: np.random.default_rng = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.default_rng(42)
        )
        self.num_arms = self.wrapped_feedback.get_num_arms()
        self.host = self.random_state.choice(self.num_arms)
        self.arm_scores = np.ones((self.num_arms,))
        self.arm_j = None
        self.i_wins = 0
        self.j_wins = 0

    def _reset_match(self):
        self.arm_j = None
        self.i_wins = 0
        self.j_wins = 0

    def step(self) -> None:
        if self.arm_j is None:
            opponent_prob = self.arm_scores / (np.sum(self.arm_scores) - self.arm_scores[self.host])
            opponent_prob[self.host] = 0
            self.arm_j = self.random_state.choice(self.num_arms, p=opponent_prob)

        arm_i, arm_j = self.host, self.arm_j
        target_rounds = self.arm_scores[self.host]

        if self.wrapped_feedback.duel(arm_i, arm_j):
            self.i_wins += 1
        else:
            self.j_wins += 1

        if self.i_wins >= target_rounds:
            self._reset_match()
            winner, loser = arm_i, arm_j
        elif self.j_wins >= target_rounds:
            self._reset_match()
            winner, loser = arm_j, arm_i
        else:
            return

        self.arm_scores[winner] += 1
        self.arm_scores[loser] = np.maximum(self.arm_scores[loser] - 1, 1)
        self.host = winner

    def get_condorcet_winner(self) -> int:
        """Get the index of the arm currently believed to be the Condorcet winner.
        """
        return self.host


