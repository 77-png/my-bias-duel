"""Utility class for tracking bias errors.
"""

from typing import Optional
import numpy as np

from duel_bias.feedback.preference_estimate_bias import PreferenceEstimateBias

class ErrorUserBias:
    """Compute the error versus a comparative user bias
    """
    def __init__(self, comparative_user_bias: np.ndarray):
        assert comparative_user_bias is not None
        assert comparative_user_bias.ndim == 1
        self.user_bias = comparative_user_bias

    def __call__(self, user_bias):
        return float(np.linalg.norm(self.user_bias - user_bias) / np.sqrt(len(user_bias)))

def compute_l2_error(x: np.ndarray, y: np.ndarray):
    """Compute L2 error between ndarray x, y
    """
    return float(np.linalg.norm(x - y) / np.sqrt(x.size))

class BiasTracker:
    """Tracks and computes bias errors during algorithm execution.
    """
    def __init__(self, comparative_user_bias: Optional[np.array] = None, compute_residual = False):
        self._compute_error_bias = ErrorUserBias(comparative_user_bias) if comparative_user_bias is not None else None
        self._is_compute_residual = compute_residual
        self.error = []
        self.residual = []
    
    def update(self, preference_estimate: PreferenceEstimateBias) -> None:
        """Update bias errors tracking.
        """
        if self._compute_error_bias is not None:
            self.error.append(
                self._compute_l2_error(preference_estimate)
            )

        if self._is_compute_residual:
            self.residual.append(
                max(
                    self._compute_residual_p(preference_estimate),
                    self._compute_residual_eta(preference_estimate)
                )
            )

    def _compute_l2_error(self, preference_estimate: PreferenceEstimateBias):
        """L2 error between estimated evaluator bias and comparative bias
        """
        return self._compute_error_bias(preference_estimate.get_user_bias())

    def _compute_residual_p(self, preference_estimate: PreferenceEstimateBias):
        """L2-Residual with respect to preference estimation
        """
        preference = preference_estimate._cached_mean_estimate
        preference2 = preference_estimate._compute_biased_mean_preference(sigma=0.0)
        return compute_l2_error(preference, preference2)

    def _compute_residual_eta(self, preference_estimate: PreferenceEstimateBias):
        """L2-Residual with respect to evaluator bias estimation
        """
        bias = preference_estimate._cached_user_bias
        bias2 = preference_estimate._compute_user_bias_from_biased_mean_preference(sigma=0.0)
        return compute_l2_error(bias, bias2)

    def get_error(self):
        return self.error, self.residual

    def get_comparative_bias(self):
        return self._compute_error_bias.user_bias if self._compute_error_bias is not None else None