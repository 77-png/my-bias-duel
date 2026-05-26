from duelpy.stats.metrics import Metric
from time import perf_counter as get_time

__all__ = [
    "HasBestArmRate",
    "RoundTime",
]

class HasBestArmRate(Metric):
    """The rate of pulling the best arm (one of is the best).
    """
    def __init__(self, best_arm: int) -> None:
        self.best_arm = best_arm

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        if arm_i_index == self.best_arm or arm_j_index == self.best_arm:
            best_arm_rate = 1.0
        else:
            best_arm_rate = 0.0
        return best_arm_rate

class RoundTime(Metric):
    """The average wall clock time for implementing each round
    """
    def __init__(self) -> None:
        super().__init__()
        self.start_time = get_time()

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        current_time = get_time()
        self.start_time, time_cost = current_time, current_time - self.start_time
        return time_cost