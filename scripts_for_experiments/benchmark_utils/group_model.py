import numpy as np
from typing import List, Union
from duelpy.stats.preference_matrix import PreferenceMatrix
from duel_bias.feedback.matrix_feedback_bias import MatrixFeedbackBias

class PreferenceGroupModelBias(MatrixFeedbackBias):
    """
    A feedback model where each arm has:
      - g_i ∈ {0, 1}, indicating group membership
      - s_i: skill or strength

    Each user has:
      - r_k: preference multiplier for arms where g_i == 1

    The comparison probability between arm i and j for user k is:
        p(i <_k j) = (r_k^{g_i} * exp(s_i)) / (r_k^{g_i} * exp(s_i) + r_k^{g_j} * exp(s_j))
    """

    def __init__(
        self,
        g: List[int],
        s: List[float],
        r: List[float],
        random_state: Union[np.random.RandomState, np.random.Generator]
    ):
        self.num_arms = len(s)
        self.num_users = len(r)
        self.g = np.array(g)
        self.s = np.array(s)
        self.r = np.array(r)
        self.random_state = random_state

        preference_matrix = PreferenceMatrix(self._construct_truth_matrix())

        # Call superclass constructor
        super().__init__(
            preference_matrix=preference_matrix,
            user_bias=np.ones(self.num_users),  # bias handled internally
            random_state=random_state
        )

    def _compute_true_probability(self, i: int, j: int) -> float:
        """Return the unbiased win probability of arm i over arm j"""
        return np.exp(self.s[i]) / (np.exp(self.s[i]) + np.exp(self.s[j]))
    
    def _construct_truth_matrix(self) -> np.ndarray:
        """Build the true (unbiased) preference matrix used for metrics"""
        matrix = np.full((self.num_arms, self.num_arms), 0.5)
        for i in range(self.num_arms):
            for j in range(self.num_arms):
                if i != j:
                    matrix[i, j] = self._compute_true_probability(i, j)
        return matrix

    def _compute_win_probability_with_bias(self, i: int, j: int, k: int = None) -> float:
        """Override to compute probability with group-sensitive user preference."""
        if k is None:
            k = self.choose_feedback_user()

        # r_k^{g_i} * e^{s_i}
        bias_i = self.r[k] if self.g[i] else 1
        bias_j = self.r[k] if self.g[j] else 1
        score_i = bias_i * np.exp(self.s[i])
        score_j = bias_j * np.exp(self.s[j])
        return score_i / (score_i + score_j)

    def get_condorcet_winner(self) -> Union[int, None]:
        """Return the Condorcet winner based on the true preference matrix (if exists)"""
        p = self.preference_matrix.preferences
        for i in range(self.num_arms):
            if all(p[i, j] > 0.5 for j in range(self.num_arms) if j != i):
                return i
        return None

    def get_best_arms(self) -> List[int]:
        """Return the Condorcet winner if exists, otherwise return the Borda winner(s)"""
        condorcet = self.get_condorcet_winner()
        if condorcet is not None:
            return [condorcet]
        scores = self.preference_matrix.preferences.sum(axis=1)
        best_score = np.max(scores)
        return [i for i, score in enumerate(scores) if score == best_score]

    def get_arbitrary_ranking(self) -> List[int]:
        """Return any valid ranking based on the true preference matrix"""
        scores = self.preference_matrix.preferences.sum(axis=1)
        return sorted(range(self.num_arms), key=lambda i: -scores[i])
    
    def test_ranking(self, ranking: List[int]) -> bool:
        """Check if the provided ranking is admissible based on s_i values (true skill)"""
        for i in range(len(ranking) - 1):
            if self.s[ranking[i]] < self.s[ranking[i + 1]]:
                return False
        return True
