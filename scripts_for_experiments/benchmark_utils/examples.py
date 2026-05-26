import numpy as np
from duel_bias.feedback.matrix_feedback_bias import BiasGenerator
from enum import Enum


# ============================================================
# 原有函数 (保持不变)
# ============================================================

def generate_task_normal(num_arms: int, loc: float = 0.0, scale: float = 1.0, seed: int = 42) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return np.sort(np.exp(generator.normal(loc=loc, scale=scale, size=num_arms)))


def get_user_bias_beta(user_num: int, alpha: float = 2.0, beta: float = 1.0, center: float = 0.5, radius: float = None) -> np.ndarray:
    if radius is None:
        radius = min(1.0 - center, center)

    if center + radius > 1.0:
        raise ValueError("center + radius should be no more than 1.0")
    elif center - radius < 0.0:
        raise ValueError("center - radius should be more than 0.0")

    generator = np.random.default_rng(42)
    return center + radius * (2 * generator.beta(alpha, beta, user_num) - 1.0)


class BetaBiasGenerator(BiasGenerator):
    def __init__(self, alpha: float, beta: float, seed: int):
        super().__init__(seed=seed)
        self.alpha = alpha
        self.beta = beta
        self.generator = None
    
    def __next__(self):
        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)
        return self.generator.beta(self.alpha, self.beta)


class UniformBiasGenerator(BiasGenerator):
    def __init__(self, low: float, high: float, seed: int):
        super().__init__(seed=seed)
        self.low = low
        self.high = high
        self.generator = None

    def __next__(self):
        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)
        return self.generator.uniform(self.low, self.high)


def get_user_bias_beta_varying(user_num: int, center: float = 0.7, variance: float = 0.1):
    _tmp = center * (1 - center) / variance - 1
    assert _tmp > 0, "The input should satisfy center * (1 - center) > variance."
    alpha = center * _tmp
    beta = (1 - center) * _tmp
    return [BetaBiasGenerator(alpha, beta, 42 + idx) for idx in range(user_num)]


def get_user_bias_uniform_varying(user_num: int, center: float = 0.7, scale: float = 0.1):
    assert center + scale <= 1.0, "center + scale should be less than 1.0"
    assert center - scale >= 0.0, "center - scale should be less than 0.0"
    return [UniformBiasGenerator(center - scale, center + scale, 42 + idx) for idx in range(user_num)]


def get_user_bias_uniform(user_num: int, center: float = 0.8, radius: float = None) -> np.ndarray:
    if radius is None:
        radius = min(1.0 - center, center)

    if center + radius > 1.0:
        raise ValueError("center + radius should be no more than 1.0")
    elif center - radius < 0.0:
        raise ValueError("center - radius should be more than 0.0")

    generator = np.random.default_rng(42)
    return generator.uniform(center - radius, center + radius, user_num)


def print_info_problem(preference, user_bias, verbose: bool = False):
    is_bias_varying = isinstance(user_bias[0], BiasGenerator)

    if not is_bias_varying:
        biased_preference = (
            preference[np.newaxis, :, :] * user_bias[:, np.newaxis, np.newaxis]
            + (1.0 - preference[np.newaxis, :, :]) * (1.0 - user_bias[:, np.newaxis, np.newaxis])
        )
    else:
        biased_preference = None

    with np.printoptions(formatter={'float_kind': lambda x: f"{x:.2f}"}):
        print("Preference Matrix:\n", preference)

        if not is_bias_varying:
            print("User Bias:\n", user_bias)

            if verbose:
                print("Biased Preference:")
                for user_idx in range(len(user_bias)):
                    print("User {}\n".format(user_idx + 1),
                          biased_preference[user_idx])


# ============================================================
# BARP风格函数 (新增)
# ============================================================

class BiasDistribution(Enum):
    """BARP偏见分布类型"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


def generate_task_barp(
    num_arms: int,
    score_mean: float = 0.0,
    score_std: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """生成BARP风格的潜在分数 s_i (log空间).
    
    Parameters
    ----------
    num_arms : int
        arm数量
    score_mean : float
        分数分布均值 (默认0.0)
    score_std : float
        分数分布标准差 (默认2.0, 对应σ²=4时用2.0, σ²=2时用sqrt(2))
    seed : int
        随机种子
        
    Returns
    -------
    np.ndarray
        潜在分数数组 s_i (log空间, 不取exp)
    """
    generator = np.random.default_rng(seed)
    return generator.normal(score_mean, score_std, size=num_arms)


def generate_group_membership(
    num_arms: int,
    group_a_ratio: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """生成组别成员关系 (BARP论文: 70/30分布).
    
    Parameters
    ----------
    num_arms : int
        arm数量
    group_a_ratio : float
        参考组比例 (默认0.7)
    seed : int
        随机种子
        
    Returns
    -------
    np.ndarray
        二值数组, 1=参考组(δ=1), 0=其他组(δ=0)
    """
    generator = np.random.default_rng(seed)
    num_group_a = int(num_arms * group_a_ratio)
    group_membership = np.zeros(num_arms, dtype=int)
    group_a_indices = generator.choice(num_arms, size=num_group_a, replace=False)
    group_membership[group_a_indices] = 1
    return group_membership

def get_user_bias_barp(num_users: int, distribution: str = "normal", 
                       mean: float = 0.0, std: float = 1.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        bias = rng.normal(mean, std, num_users)
        bias = np.clip(bias, -6.0, 6.0)
    elif distribution == "beta":
        alpha, beta_param = 2.0, 1.0
        bias = rng.beta(alpha, beta_param, num_users) * 12 - 6
    else:
        raise ValueError("distribution must be 'normal' or 'beta'")
    return bias.astype(float)

# def get_user_bias_barp(
#     user_num: int,
#     distribution: str = "normal",
#     mean: float = 0.0,
#     std: float = 1.0,
#     min_val: float = -5.0,
#     max_val: float = 5.0,
#     seed: int = 42,
# ) -> np.ndarray:
#     """生成BARP风格的用户偏见参数 θ_k.
    
#     BARP偏见模型: P(i >_k j) = exp(s_i + θ_k*δ_i) / (exp(s_i + θ_k*δ_i) + exp(s_j + θ_k*δ_j))
    
#     Parameters
#     ----------
#     user_num : int
#         用户数量
#     distribution : str
#         分布类型: 'normal', 'uniform', 'exponential'
#     mean : float
#         Normal分布均值 (默认0.0)
#     std : float
#         Normal分布标准差 (默认1.0), 也用于Exponential的scale
#     min_val : float
#         Uniform分布最小值 (默认-5.0)
#     max_val : float
#         Uniform分布最大值 (默认5.0)
#     seed : int
#         随机种子
        
#     Returns
#     -------
#     np.ndarray
#         θ_k数组:
#         - θ_k > 0: 偏向参考组 (δ=1的arm)
#         - θ_k < 0: 偏向非参考组 (δ=0的arm)
#         - θ_k = 0: 无偏见
    
#     Examples
#     --------
#     # BARP Figure 2 Left: 增加偏见均值
#     >>> bias = get_user_bias_barp(10, 'normal', mean=2.0, std=1.0)
    
#     # BARP Figure 2 Right: 增加偏见方差
#     >>> bias = get_user_bias_barp(10, 'normal', mean=0.0, std=3.0)
    
#     # BARP Figure 3 Right: Uniform [-5, 5]
#     >>> bias = get_user_bias_barp(10, 'uniform', min_val=-5.0, max_val=5.0)
#     """
#     generator = np.random.default_rng(seed)
#     distribution = distribution.lower()
    
#     if distribution == "normal":
#         return generator.normal(mean, std, size=user_num)
#     elif distribution == "uniform":
#         return generator.uniform(min_val, max_val, size=user_num)
#     elif distribution == "exponential":
#         values = generator.exponential(std, size=user_num)
#         signs = generator.choice([-1, 1], size=user_num)
#         return values * signs
#     else:
#         raise ValueError(f"Unknown distribution: {distribution}. Use 'normal', 'uniform', or 'exponential'.")


class NormalBiasGenerator(BiasGenerator):
    """BARP风格的Normal分布θ_k生成器"""
    def __init__(self, mean: float, std: float, seed: int):
        super().__init__(seed=seed)
        self.mean = mean
        self.std = std
        self.generator = None
    
    def __next__(self):
        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)
        return self.generator.normal(self.mean, self.std)


class UniformThetaBiasGenerator(BiasGenerator):
    """BARP风格的Uniform分布θ_k生成器 (范围可以是负数)"""
    def __init__(self, min_val: float, max_val: float, seed: int):
        super().__init__(seed=seed)
        self.min_val = min_val
        self.max_val = max_val
        self.generator = None

    def __next__(self):
        if self.generator is None:
            self.generator = np.random.default_rng(self.seed)
        return self.generator.uniform(self.min_val, self.max_val)


def get_user_bias_barp_varying(
    user_num: int,
    distribution: str = "normal",
    mean: float = 0.0,
    std: float = 1.0,
    min_val: float = -5.0,
    max_val: float = 5.0,
):
    """获取BARP风格的动态偏见生成器列表.
    
    每个用户有独立的生成器, 每次采样产生不同的θ_k值.
    """
    distribution = distribution.lower()
    
    if distribution == "normal":
        return [NormalBiasGenerator(mean, std, 42 + idx) for idx in range(user_num)]
    elif distribution == "uniform":
        return [UniformThetaBiasGenerator(min_val, max_val, 42 + idx) for idx in range(user_num)]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def print_info_problem_barp(
    preference: np.ndarray,
    user_bias: np.ndarray,
    group_membership: np.ndarray = None,
    skill_vector: np.ndarray = None,
    verbose: bool = False,
):
    """打印BARP风格实验信息.
    
    Parameters
    ----------
    preference : np.ndarray
        偏好矩阵
    user_bias : np.ndarray
        用户偏见θ_k数组
    group_membership : np.ndarray, optional
        组别成员关系
    skill_vector : np.ndarray, optional
        潜在分数s_i
    verbose : bool
        是否打印详细信息
    """
    is_bias_varying = isinstance(user_bias[0], BiasGenerator)
    
    print("=" * 60)
    print("BARP-Style Experiment Info")
    print("=" * 60)
    print(f"Number of arms: {preference.shape[0]}")
    print(f"Number of users: {len(user_bias)}")
    
    if skill_vector is not None:
        print(f"\nSkill vector (s_i) - log space:")
        print(f"  Range: [{skill_vector.min():.3f}, {skill_vector.max():.3f}]")
        print(f"  Mean: {skill_vector.mean():.3f}, Std: {skill_vector.std():.3f}")
    
    if group_membership is not None:
        num_group_a = (group_membership == 1).sum()
        num_group_b = (group_membership == 0).sum()
        print(f"\nGroup distribution:")
        print(f"  Reference group (δ=1): {num_group_a} ({100*num_group_a/len(group_membership):.1f}%)")
        print(f"  Other group (δ=0): {num_group_b} ({100*num_group_b/len(group_membership):.1f}%)")
    
    if not is_bias_varying:
        print(f"\nUser bias (θ_k):")
        print(f"  Range: [{user_bias.min():.3f}, {user_bias.max():.3f}]")
        print(f"  Mean: {user_bias.mean():.3f}, Std: {user_bias.std():.3f}")
        print(f"  θ>0 (favor ref group): {(user_bias > 0).sum()}")
        print(f"  θ<0 (favor other group): {(user_bias < 0).sum()}")
        print(f"  θ≈0 (|θ|<0.1): {(np.abs(user_bias) < 0.1).sum()}")
        
        if verbose:
            with np.printoptions(formatter={'float_kind': lambda x: f"{x:.3f}"}):
                print(f"  Values: {user_bias}")
    else:
        print(f"\nUser bias: Dynamic (BiasGenerator)")
    
    print("=" * 60)