# 2025-05-04 Distributed value sampler

import numpy as np
import matplotlib.pyplot as plt


# Not used
def detect_gaps(a_unique: np.array, n: int) -> bool:
    '''heuristic to avoid empty bins'''
    bin_width = (a_unique.max() - a_unique.min()) / (n - 1)
    gap_detected = False
    dist = 0
    for i in range(len(a_unique)-1):
        prev_dist = dist
        dist = a_unique[i+1] - a_unique[i]
        if dist + prev_dist > 2 * bin_width:
            gap_detected = True
    return gap_detected


def prepare_array(arr: np.array, n: int):
    arr_sorted = np.sort(arr)
    arr_unique = np.unique(arr)
    if len(arr_unique) < n:
        err_msg = f'N={n} is higher than the number of unique values.'
        raise ValueError(err_msg)
    return arr_unique, arr_sorted


def prepare_bins(arr_unique: np.array, n: int) -> tuple:
    min_ = arr_unique.min()
    max_ = arr_unique.max()
    a_linspace = np.linspace(
        min_,
        max_,
        n * 2 - 1
    )
    bin_borders = a_linspace[1::2]
    bin_borders = np.concatenate([[min_], bin_borders, [max_]])
    bin_centers = a_linspace[::2]
    bins = {}
    for i in range(len(bin_borders)-1):
        bins[i] = {
            'left': bin_borders[i],
            'right': bin_borders[i+1],
            'center': bin_centers[i],
            'values': []
        }
    return bins, bin_borders, bin_centers


def plot(a_unique: np.array, bin_borders: np.array, bin_centers: np.array):
    plt.figure(figsize=(10, 1))
    #unique values
    plt.scatter(
        x=a_unique,
        y=np.ones_like(a_unique),
        c='gray'
    )
    #bin borders
    plt.scatter(
        x=bin_borders[1:-1],
        y=np.ones_like(bin_borders[1:-1]),
        marker='|',
        s=2000,
        c='orange'
    )
    # bin centers
    plt.scatter(
        x=bin_centers,
        y=np.ones_like(bin_centers),
        marker='x',
        s=20,
        c='red'
    )
    plt.yticks([])
    plt.show()
    return None


def split_into_bins(bins: dict, arr_sorted: np.array) -> dict:    
    n_bins = len(bins)
    current_bin = 0
    for val in arr_sorted:
        val_is_in_bin = False
        # print(f'{val=}', f'{current_bin=}')
        if val < bins[current_bin]['left']:
            continue
        while not val_is_in_bin:
            # print(f'{current_bin=}', bins[current_bin]['left'], bins[current_bin]['right'])
            val_is_in_bin = bins[current_bin]['left'] <= val
            val_is_in_bin &= bins[current_bin]['right'] > val
            if not val_is_in_bin:
                current_bin += 1
            if current_bin >= n_bins:
                break
        if current_bin >= n_bins:
            break
        else:
            bins[current_bin]['values'].append(val)
    bins[current_bin - 1]['values'].append(arr_sorted[-1])
    return bins


def prepare_bins_n_observations(arr: np.array, n: int, show_plot: bool = False) -> dict:
    okay = False
    n_adjusted = n
    count = 0
    arr_unique, arr_sorted = prepare_array(arr, n)
    while not okay:
        count += 1
        bins, bin_borders, bin_centers = prepare_bins(arr_unique, n_adjusted)
        # plot(a_unique, bin_borders, bin_centers)
        bins = split_into_bins(bins, arr_sorted)
        count_nonempty_bins = np.sum([1 for bin in bins.values() if bin['values']])
        if count_nonempty_bins < n:
            n_adjusted += 1
        else:
            okay = True
        if count == len(arr_unique):
            break
    if show_plot:
        plot(arr_unique, bin_borders, bin_centers)
    return bins


def get_selection_sequence(bins: dict) -> np.array:
    selection_sequence = []
    for bin in bins.values():
        bin_len = len(bin['values'])
        if bin_len:
            if bin_len == 1:
                selection_sequence.append([1])
            else:
                central_value_idx = 0
                val_0 = bin['values'][central_value_idx]
                min_dist_center = np.abs(val_0 - bin['center'])      
                for bin_val_idx in range(1, bin_len):
                    val = bin['values'][bin_val_idx]
                    dist_center = np.abs(val - bin['center'])
                    # print(f"{val=} center={bin['center']} {dist_center=} left={bin['left']} right={bin['right']}")
                    if dist_center < min_dist_center:
                        min_dist_center = dist_center
                        central_value_idx = bin_val_idx
                bin_map = np.zeros(bin_len)
                bin_map[central_value_idx] = 1
                selection_sequence.append(bin_map)
    selection_sequence = np.concatenate(selection_sequence).astype(bool)
    return selection_sequence


def select_by_value(arr: np.array, n: int, show_plot: bool = False):
    bins = prepare_bins_n_observations(arr, n, show_plot=show_plot)
    selection_sequence = get_selection_sequence(bins)
    return selection_sequence

