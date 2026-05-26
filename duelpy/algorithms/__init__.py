"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.approximate_probability import ApproximateProbability
from duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelpy.algorithms.beat_the_mean import BeatTheMeanBanditPAC
from duelpy.algorithms.binary_search_ranking import BinarySearchRanking
from duelpy.algorithms.borda_ranking import BordaRanking
from duelpy.algorithms.copeland_confidence_bound import CopelandConfidenceBound
from duelpy.algorithms.cw_rmed import CwRmed
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSamplingPlus
from duelpy.algorithms.exploreverify import VerificationBasedCondorcet
from duelpy.algorithms.interleaved_filtering import InterleavedFiltering
from duelpy.algorithms.kl_divergence_based_pac import KLDivergenceBasedPAC
from duelpy.algorithms.knockout_tournament import KnockoutTournament
from duelpy.algorithms.mallows import MallowsMPI
from duelpy.algorithms.mallows import MallowsMPR
from duelpy.algorithms.merge_rucb import MergeRUCB
from duelpy.algorithms.multisort import Multisort
from duelpy.algorithms.optmax import OptMax
from duelpy.algorithms.plackett_luce import PlackettLuceAMPR
from duelpy.algorithms.plackett_luce import PlackettLucePACItem
from duelpy.algorithms.relative_confidence_sampling import RelativeConfidenceSampling
from duelpy.algorithms.relative_ucb import RelativeUCB
from duelpy.algorithms.rmed import Rmed1
from duelpy.algorithms.rmed import Rmed2
from duelpy.algorithms.rmed import Rmed2FH
from duelpy.algorithms.savage import Savage
from duelpy.algorithms.scalable_copeland_bandits import ScalableCopelandBandits
from duelpy.algorithms.sequential_elimination import SequentialElimination
from duelpy.algorithms.single_elimination_tournament import SingleEliminationTop1Select
from duelpy.algorithms.single_elimination_tournament import SingleEliminationTopKSorting
from duelpy.algorithms.successive_elimination import SuccessiveElimination
from duelpy.algorithms.winner_stays import WinnerStaysStrongRegret
from duelpy.algorithms.winner_stays import WinnerStaysWeakRegret


# Pylint insists that regret_minimizing_algorithms and interfaces are constants and should be
# named in UPPER_CASE. Technically that is correct, but it doesn't feel quite
# right for this use case. Its not a typical constant. A similar use-case would
# be numpy's np.core.numerictypes.allTypes, which is also not names in
# UPPER_CASE.
# pylint: disable=invalid-name

# Make the actual algorithm classes available for easy enumeration in
# experiments and tests.
# All algorithms that include some sort of regret-minimizing mode. That
# includes PAC algorithms with an (optional) exploitation phase.
regret_minimizing_algorithms = [
    Savage,
    WinnerStaysWeakRegret,
    WinnerStaysStrongRegret,
    BeatTheMeanBandit,
    BeatTheMeanBanditPAC,
    RelativeConfidenceSampling,
    RelativeUCB,
    InterleavedFiltering,
    KnockoutTournament,
    CopelandConfidenceBound,
    MallowsMPI,
    MallowsMPR,
    Multisort,
    MergeRUCB,
    SequentialElimination,
    SingleEliminationTopKSorting,
    SingleEliminationTop1Select,
    SuccessiveElimination,
    DoubleThompsonSampling,
    DoubleThompsonSamplingPlus,
    PlackettLucePACItem,
    PlackettLuceAMPR,
    OptMax,
    ScalableCopelandBandits,
    KLDivergenceBasedPAC,
    VerificationBasedCondorcet,
    BinarySearchRanking,
    BordaRanking,
    CwRmed,
    Rmed1,
    Rmed2,
    Rmed2FH,
]
other_algorithms = [ApproximateProbability]
# This is not really needed, but otherwise zimports doesn't understand the
# __all__ construct and complains that the Algorithm import is unnecessary.
interfaces = [Algorithm]

# Generate __all__ for tab-completion etc.
__all__ = (
    ["Algorithm"]
    + [algorithm.__name__ for algorithm in regret_minimizing_algorithms]
    + [algorithm.__name__ for algorithm in other_algorithms]
)
