"""Algorithms for solving Dueling Bandit problems.
"""

from duel_bias.algorithms.ucb import UCB
from duel_bias.algorithms.beat_the_winner import BeatTheWinner, ModifiedBeatTheWinner

from duel_bias.algorithms.relative_ucb_bias import RelativeUCBBias
from duel_bias.algorithms.relative_confidence_sampling_bias import RelativeConfidenceSamplingBias
from duel_bias.algorithms.double_thompson_sampling_bias import DoubleThompsonSamplingBias
from duel_bias.algorithms.ucb_bias import BiasSensitiveUCB

__all__ = [
    "UCB",
    "BeatTheWinner",
    "ModifiedBeatTheWinner",
    "RelativeUCBBias",
    "RelativeConfidenceSamplingBias",
    "DoubleThompsonSamplingBias",
    "BiasSensitiveUCB",
]