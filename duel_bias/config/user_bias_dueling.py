"""Config for the dueling model with biased evaluator.
"""

class DefaultConfig:
    """Default config for the dueling model with user bias

    Parameters
    ----------
    seed
        random seed
    piecewise_count_max
        select a new user each <piecewise_count_max> rounds
    """
    seed = 42
    piecewise_count_max = 0

class UserEvenly(DefaultConfig):
    """Select user evenly
    """
    user_participation = 'evenly'

class UserBeta(DefaultConfig):
    """Set user participation as beta distribution with parameters alpha and beta
    """
    user_participation = 'beta'
    alpha = 2.0
    beta = 2.0

# Choose the settings for the dueling model with user bias
Config = UserEvenly
