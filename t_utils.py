import numpy as np
from itertools import groupby


def integrate_compressed(compressed):
    unique_values, end_indices = compressed
    unique_values = np.clip(unique_values, a_min=0, a_max=None)

    condition = (unique_values > 0.9999) & (unique_values < 10000)

    condition[condition] = 0

    total_sum = 0
    start_idx = 0

    for value, end_idx in zip(unique_values, end_indices):
        count = end_idx - start_idx + 1
        total_sum += value * count
        start_idx = end_idx + 1

    return total_sum


def compress_array(arr):
    unique_values = []
    end_indices = []
    arr = np.round(arr, 5)
    for key, group in groupby(enumerate(arr), key=lambda x: x[1]):
        group = list(group)
        end_idx = group[-1][0]
        unique_values.append(key)
        end_indices.append(end_idx)
    return [np.array(unique_values), np.array(end_indices)]


def get_slope_accelerated(location, slope_seg, slope, train_info):
    """
        Train slope acceleration
        Parameters:
        - location: current position of the train
        - slope_seg: list of slope segments
        - slope: list of slope values corresponding to segments
        - train_info: dictionary containing train details (length, mass, etc.)

        Returns:
        - acc: train slope acceleration
    """
    train_len = train_info['len']
    slope_accelerated = 0.0
    G = 9.81  # gravitational acceleration

    for section in range(len(slope_seg)):
        # If the train is within this slope segment (a,b]
        if slope_seg[section] < location <= slope_seg[section + 1]:
            # If the train's position is within the train length in the current segment
            if location <= train_len + slope_seg[section]:
                if section == 0:
                    slope_accelerated = (slope[section] * G * location) / (train_len * 1000)
                    break
                else:
                    slope_accelerated = (slope[section] * G * (location - slope_seg[section])) / (train_len * 1000) \
                                        + (slope[section - 1] * G * (train_len + slope_seg[section] - location)) / (
                                                    train_len * 1000)
                    break
            # If the position exceeds the segment within the train length
            if location > train_len + slope_seg[section]:
                slope_accelerated = slope[section] * G / 1000
                break

    return slope_accelerated


def get_line_action(line, slope_seg, slope, train_info):
    """
    Get action values at each position
    Parameters:
    - line (list): list of speed values along the track
    - slope_seg: list of slope segments
    - slope: list of slope values corresponding to segments
    - train_info: dictionary containing train details (traction, power, mass, etc.)

    Returns:
    - list: list of action values for each position
    """
    max_traction = train_info['traction_140']
    train_power = train_info['power']
    mass = train_info['mass']
    max_braking = train_info['braking_140']
    mass_factor = 1 + train_info['mass_factor']

    actions = [0]
    for loc in range(len(line) - 1):
        loc += 1  # update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # get slope acceleration
        last_v = line[loc - 1]  # get the previous speed value
        basic_acc = (train_info['drag_coefficient_a'] + train_info['drag_coefficient_b'] * last_v * 3.6
                     + train_info['drag_coefficient_c'] * (last_v * 3.6) ** 2) / train_info['mass']
        non_gear_acc = -(slope_acc + basic_acc)
        if line[loc - 1] != 0:
            max_gear_action = min(max_traction, train_power / (mass * mass_factor * line[loc - 1]))
        else:
            max_gear_action = max_traction
        gear_acc = out_acc - non_gear_acc
        if gear_acc >= 0:
            action = min(gear_acc / max_gear_action, 1)
        else:
            action = max(-gear_acc / max_braking, -1)
        actions.append(action)
    return np.array(actions)


def get_line_energy(line, slope_seg, slope, train_info):
    """
    Get energy values for each position
    Parameters:
    - line (list): list of speed values along the track
    - slope_seg: list of slope segments
    - slope: list of slope values corresponding to segments
    - train_info: dictionary containing train details

    Returns:
    - float: total energy consumption
    """
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    c_1 = train_info['drag_coefficient_a']
    c_2 = train_info['drag_coefficient_b']
    c_3 = train_info['drag_coefficient_c']
    mass = train_info['mass']
    e = 0

    for loc in range(len(line) - 1):
        loc += 1  # update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # get slope acceleration
        last_v = line[loc - 1]  # get the previous speed value
        basic_acc = (c_1 + c_2 * last_v * 3.6 + c_3 * (last_v * 3.6) ** 2) / mass
        non_gear_acc = -(basic_acc + slope_acc)

        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # convert to kWh
            e += (gear_acc * mass) / 3600

    return e


def get_line_step_time(line):
    """
    Get time values at each step based on the speed profile
    Parameters:
    - line (list): list of speed values along the track

    Returns:
    - list: list of cumulative time values at each step
    """
    ts = [0]
    t = 0
    for loc in range(1, len(line) + 1):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0 + v1 == 0")
        t += 2 / (line[loc - 1] + line[loc])
        ts.append(t)
    ts = np.array(ts)
    return ts


def get_line_step_energy(line, slope_seg, slope, train_info):
    """
    Get energy values at each step
    Parameters:
    - line (list): list of speed values along the track
    - slope_seg: list of slope segments
    - slope: list of slope values corresponding to segments
    - train_info: dictionary containing train details

    Returns:
    - list: list of cumulative energy values at each step
    """
    es = [0]
    e = 0
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']

    for loc in range(len(line) - 1):
        loc += 1  # update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # get slope acceleration
        last_v = line[loc - 1]  # get the previous speed value
        non_gear_acc = -(slope_acc + (train_info['drag_coefficient_a'] + train_info['drag_coefficient_b'] * last_v * 3.6
                                      + train_info['drag_coefficient_c'] * (last_v * 3.6) ** 2) / train_info[
                             'mass'])  # calculate acceleration
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # convert to kWh
            e += (gear_acc * mass) / 3600
        es.append(e)

    return np.array(es)
