from typing import Optional
from types import SimpleNamespace
import numpy as np
from collections import defaultdict

from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate
from duelpy.stats.preference_matrix import PreferenceMatrix

import duel_bias.config.bias_estimation
import duel_bias.config.preference_estimation

class GroupEstimateBias(PreferenceEstimate):
    """Estimate the group bias of a user based on their feedback.

    This class estimates the group bias for each user based on their feedback
    and the group membership of the arms they interact with.
    """

    def __init__(
        self,
        num_arms: int,
        num_users: int,
        confidence_radius=None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        start_update_theta_after: int = 20,
        buck = 50,
    ) -> None:
        super().__init__(num_arms, confidence_radius)
        
        self.num_users = num_users
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.start_update_theta_after = start_update_theta_after
        
        # Initialize parameters
        self.s = np.random.randn(num_arms)  # Item strengths
        self.theta = np.random.randn(num_users)  # User biases
        self.g = np.random.randint(0, 2, size=num_arms)  # Item groups (0 or 1)
        
        # Momentum terms
        self.momentum_s = np.zeros(num_arms)
        self.momentum_theta = np.zeros(num_users)
        
        # Track counts per user
        self.user_counts = defaultdict(int)  # (i,j,k) -> wins of i over j by user k
        self.user_totals = defaultdict(int)  # (i,j,k) -> total comparisons between i,j by user k
        
        # Track rounds for theta update
        self.round = 0

        self._cached_mean_estimate = np.zeros((self.num_arms, self.num_arms))
        self.buck = buck

    def enter_sample(
        self, 
        first_arm_index: int, 
        second_arm_index: int, 
        first_won: bool,
        user_index: Optional[int] = None
    ) -> None:
        """Enter the result of a sampled duel, optionally with user information.
        
        Parameters
        ----------
        first_arm_index : int
            The index of the first arm of the duel.
        second_arm_index : int
            The index of the second arm of the duel.
        first_won : bool
            Whether the first arm won the duel.
        user_index : Optional[int]
            The index of the user who performed the comparison. If None, uses standard update.
        """
        if user_index is None:
            # Fall back to standard preference estimate if no user info
            super().enter_sample(first_arm_index, second_arm_index, first_won)
            return
            
        # Update counts for biased model
        if first_won:
            self.user_counts[(first_arm_index, second_arm_index, user_index)] += 1
        else:
            self.user_counts[(second_arm_index, first_arm_index, user_index)] += 1
            
        self.user_totals[(first_arm_index, second_arm_index, user_index)] += 1
        self.user_totals[(second_arm_index, first_arm_index, user_index)] += 1
        
        # Update parameters
        if self.round % self.buck == 0:
            self._update_parameters()
        
        # Update round counter
        self.round += 1
    
    def get_user_bias(self) -> float:
        return self.theta
    
    def get_biased_preference_matrix(self):
        user_preference = np.zeros((self.num_users, self.num_arms, self.num_arms))
        for k in range(self.num_users):
            for i in range(self.num_arms):
                for j in range(self.num_arms):
                    if i == j:
                        user_preference[k, i, j] = 0.5
                    else:
                        si, sj = self.s[i], self.s[j]
                        gi, gj = self.g[i], self.g[j]
                        th = self.theta[k]
                        numerator = np.exp(si + th * gi)
                        denominator = numerator + np.exp(sj + th * gj)
                        user_preference[k, i, j] = numerator / denominator
        return user_preference

    def _compute_biased_mean_preference(self, sigma: float = 0.0) -> np.ndarray:
        """Compute the mean preference matrix with group-sensitive user bias."""

        user_preference = self.get_biased_preference_matrix()
        weight_bias = self.num_users * (2 * self.theta - 1.0) 
        temp_value = user_preference + self.theta[:, np.newaxis, np.newaxis] - 1.0 
        weighted_sum = np.sum(temp_value / weight_bias[:, np.newaxis, np.newaxis], axis=0)
        result = np.clip(weighted_sum, 0.0, 1.0)

        return result
    
    def _update_parameters(self) -> None:
        """Update s, theta, and g parameters based on current counts."""
        # Update group assignments
        self._update_group_assignments()
        
        # Determine whether to update theta
        update_theta = self.round >= self.start_update_theta_after
        
        # Reset momentum
        self.momentum_s.fill(0)
        if update_theta:
            self.momentum_theta.fill(0)
        
        # Perform gradient steps
        for _ in range(10):  # num_step = 10
            grad_s = np.zeros(self.num_arms)
            if update_theta:
                grad_theta = np.zeros(self.num_users)
            
            # Compute gradients
            for (i, j, k), n_ij in self.user_totals.items():
                if i == j or n_ij == 0:
                    continue
                    
                freq = self.user_counts[(i, j, k)] / n_ij
                si, sj = self.s[i], self.s[j]
                gi, gj = self.g[i], self.g[j]
                th = self.theta[k]
                
                exp1 = np.exp(si + th * gi)
                exp2 = np.exp(sj + th * gj)
                pred = exp1 / (exp1 + exp2)
                err = pred - freq
                
                grad_s[i] += err * pred * (1 - pred)
                grad_s[j] -= err * pred * (1 - pred)
                
                if update_theta:
                    grad_theta[k] += err * pred * (1 - pred) * (gi - gj)
            
            # Update momentum and parameters
            self.momentum_s = self.momentum * self.momentum_s + (1 - self.momentum) * grad_s
            self.s -= self.learning_rate * self.momentum_s
            
            if update_theta:
                self.momentum_theta = self.momentum * self.momentum_theta + (1 - self.momentum) * grad_theta
                self.theta -= self.learning_rate * self.momentum_theta
        
        self._cached_mean_estimate = np.zeros((self.num_arms, self.num_arms))

        for i in range(self.num_arms):
            for j in range(self.num_arms):
                if i == j:
                    self._cached_mean_estimate[i, j] = 0.5
                else:
                    si, sj = self.s[i], self.s[j]
                    self._cached_mean_estimate[i, j] = np.exp(si) / (np.exp(si) + np.exp(sj))
    
    def _update_group_assignments(self) -> None:
        """Update group assignments based on current parameters."""
        new_g = self.g.copy()
        
        for i in range(self.num_arms):
            losses = []
            
            for trial_gi in [0, 1]:
                g_trial = self.g.copy()
                g_trial[i] = trial_gi
                loss = 0.0
                
                for k in range(self.num_users):
                    for j in range(self.num_arms):
                        if i == j:
                            continue
                            
                        if (i, j, k) not in self.user_totals:
                            continue
                            
                        gi, gj = g_trial[i], g_trial[j]
                        gamma_i = np.exp(self.theta[k] * gi)
                        gamma_j = np.exp(self.theta[k] * gj)
                        s_i, s_j = self.s[i], self.s[j]
                        
                        pk = gamma_i * np.exp(s_i) / (gamma_i * np.exp(s_i) + gamma_j * np.exp(s_j))
                        observed = self.user_counts[(i, j, k)] / (self.user_totals[(i, j, k)] + 1e-8)
                        loss += (pk - observed) ** 2
                
                losses.append(loss)
            
            new_g[i] = 0 if losses[0] < losses[1] else 1
        
        self.g = new_g
