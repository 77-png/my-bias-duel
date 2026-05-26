import numpy as np

def argmax_set_1(x: np.ndarray) -> int:
    """Return the index of the largest element in the array.

    If there are multiple largest elements, return a random one of them.

    Args:
        x (ndarray): Array of values.

    Returns:
        ndarray: Index of the largest element in the array.
    """
    return int(np.random.choice(np.where(x == x.max())[0]))

def argmax_set_2(x: np.ndarray) -> list[int]:
    """Return the index of the largest two elements in the array.

    If there are multiple largest elements, return two random items of them.

    Args:
        x (ndarray): Array of values.

    Returns:
        ndarray: Index of the largest two elements in the array.
    """
    max_indices = np.where(x == x.max())[0]
    if len(max_indices) >= 2:
        return np.random.choice(max_indices, size=2, replace=False)
    else:
        second_max = np.max(x[x != x.max()])
        second_indices = np.where(x == second_max)[0]
        return [int(max_indices[0]), int(np.random.choice(second_indices))]
