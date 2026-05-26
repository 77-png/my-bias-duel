"""Implementation of the base class for bias-aware algorithms."""

from typing import Optional
from duelpy.algorithms.algorithm import Algorithm
from duel_bias.feedback.bias_tracker import BiasTracker
from duel_bias.feedback import PreferenceEstimateBias

class BiasAlgorithm(Algorithm):
    """Base class for all bias-aware algorithms.
    
    This class extends the base Algorithm class to provide additional support for
    bias-aware algorithms, including bias tracking and error computation.
    """
    
    def __init__(self, kwargs_bias_tracker, *args, **kwargs):
        """Initialize BiasAlgorithm.
        
        All arguments are passed to the parent Algorithm class.
        """
        super().__init__(*args, **kwargs)
        self.bias: Optional[list[float]] = None # Estimated Evaluator Bias
        self.comparative_bias: Optional[list[float]] = None # Comparative Evaluator Bias
        self.bias_tracker = BiasTracker(**kwargs_bias_tracker)
        self.preference_estimate: Optional[PreferenceEstimateBias] = None
    
    def run(self) -> None:
        """Run the algorithm until completion.
        
        This method extends the base run() method to provide additional support for
        bias tracking and error computation. It ensures that bias-related metrics
        are properly tracked throughout the algorithm's execution.
        """
        while not self.is_finished():
            try:
                self.step()
                # Update bias tracking metrics if available
                if hasattr(self, 'bias_tracker') and isinstance(self.bias_tracker, BiasTracker):
                    self.bias_tracker.update(self.preference_estimate)
            except self.wrapped_feedback.exception_class:
                # Duel budget exhausted
                pass

        self.bias = self.preference_estimate.get_user_bias()
        self.comparative_bias = self.bias_tracker.get_comparative_bias()


    def get_bias(self):
        """Get the evaluator bias
        """
        return self.bias

    def get_comparative_bias(self):
        """Get the evaluator comparative bias
        """
        return self.comparative_bias