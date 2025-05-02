# 2024-10-15
import numpy as np


def slice_mass(sorted_array, percentage):
    total_mass = np.sum(sorted_array)
    target_mass = percentage * total_mass
    current_mass = 0
    cutoff_index = 0

    for i, value in enumerate(sorted_array):
        current_mass += value
        if current_mass >= target_mass:
            cutoff_index = i
            break

    return sorted_array[:cutoff_index + 1]


def normalize(
    array: np.array,
    min_: float = None,
    max_: float = None
):
    if len(np.unique(array)) == 1:
        value = np.unique(array)[0]
        norm_value = 0.5
        if max_ != min_:
            norm_value = (value - min_) / (max_ - min_)
        return np.ones_like(array) * norm_value
    if min_ is None:
        min_ = array.min()
    if max_ is None:
        max_ = array.max()
    return (array - min_) / (max_ - min_)


def least_squares(x: np.array, y: np.array):
    '''
    returns a, b for y = ax + b with x and y series given.
    '''
    n = len(x)    
    if n == 0:
        return None, None
    elif n == 1:
        return None, y[0]
    elif n == 2:
        a = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - a * x[0]
        return a, b
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)    
    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        return None, None
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator    
    return a, b


def hole_metrics(arr) -> tuple:
    '''
    fraction, average_length, frequency = hole_metrics(arr)
    '''
    total_length = len(arr)
    if not total_length:
        return None, None, None    
    zero_mask = arr == 0    
    hole_fraction = np.sum(zero_mask) / total_length
    if not hole_fraction:
        return 0, 0, 0    
    zero_diff = np.diff(zero_mask.astype(int))
    start_indices = np.where(zero_diff == 1)[0] + 1
    end_indices = np.where(zero_diff == -1)[0]    
    if zero_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)    
    if len(start_indices) > len(end_indices):
        end_indices = np.append(end_indices, total_length - 1)    
    hole_lengths = end_indices - start_indices + 1
    hole_count = len(start_indices)
    average_length = np.mean(hole_lengths)
    frequency = hole_count / total_length    
    return hole_fraction, average_length, frequency


def center_of_mass(arr):
    len_ = len(arr)
    if not len_:
        return None
    if len_ == 1:
        return 0.5
    total_mass = np.sum(arr)    
    cumulative_mass = np.cumsum(arr)    
    com_index = np.searchsorted(cumulative_mass, total_mass / 2)    
    current_value = arr[com_index]    
    left_mass = cumulative_mass[com_index - 1]
    right_mass = total_mass - left_mass - current_value
    difference = right_mass - left_mass
    k = 0.5 * (1 + difference / current_value)
    step = 1 / len_    
    start_position = com_index / len_
    com = start_position + step * k
    return com


def compute_distribution_metrics(arr: np.array):
    len_ = len(arr)
    hole_fraction, hole_average_length, hole_frequency = hole_metrics(arr)    
    arr_normalized = normalize(arr)
    slope, elevation = least_squares(
        normalize(np.arange(len_)),
        arr_normalized
    )
    distribution_metrics = {
        'len': len_,
        'hole_fraction': hole_fraction,
        'hole_average_length': hole_average_length,
        'hole_frequency': hole_frequency,
        'center_of_mass_sorted': center_of_mass(sorted(arr)),
        'std_normalized': arr_normalized.std(),
        'slope_normalized': slope,
        'elevation_normalized': elevation,
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'median': np.median(arr),
        'mean_positive': arr[np.where(arr > 0)].mean()
    }
    return distribution_metrics