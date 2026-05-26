"""Dueling Model with user bias
"""

import numpy as np
from duel_bias.config.user_bias_dueling import Config

class UserBiasDueling:
    """Dueling Model with user bias
    """

    def __init__(self, num_users):
        # Read config
        if hasattr(Config, 'seed'):
            seed = Config.seed
        else:
            seed = 42

        if hasattr(Config, 'user_participation'):
            user_participation = Config.user_participation
        else:
            user_participation = 'evenly'

        self.num_users = num_users
        self.rng = np.random.default_rng(seed)

        # Choose a user to give feedback at current round
        match user_participation:
            case 'evenly':
                self.choose_feedback_user = self._select_evenly()
            case 'beta':
                if hasattr(Config, 'alpha') and hasattr(Config, 'beta'):
                    alpha = Config.alpha
                    beta = Config.beta
                else:
                    alpha = 2.0
                    beta = 2.0

                self.choose_feedback_user = self._select_in_beta_distribution(alpha, beta)
            case _:
                raise ValueError(f"Unknown user_participation: {user_participation}")

        # Choose user each round or each N rounds?
        if hasattr(Config, 'piecewise_count_max'):
            self.piecewise_count_max = Config.piecewise_count_max
        else:
            self.piecewise_count_max = 0  # Select a new user each round

        if self.piecewise_count_max > 0:
            self.piecewise_count = np.inf
            self.selected_user = None
            self.choose_feedback_user = self._select_piecewise(self.choose_feedback_user)

    def _select_evenly(self):
        """Evenly select user
        """
        def select_user():
            return self.rng.choice(self.num_users)
        return select_user

    def _select_in_beta_distribution(self, alpha: float = 2.0, beta: float = 2.0):
        """Select user in beta distribution
        """
        def select_user():
            return min(int(self.rng.beta(alpha, beta) * self.num_users), self.num_users - 1)
        return select_user

    def _select_piecewise(self, select_user):
        """Decorator for selecting user every <self.piecewise_count_max> rounds
        """
        def wrapper(*args, **kwargs):
            if self.piecewise_count >= self.piecewise_count_max:
                self.piecewise_count = 0
                self.selected_user = select_user(*args, **kwargs)
            else:
                self.piecewise_count += 1
            return self.selected_user
        return wrapper
