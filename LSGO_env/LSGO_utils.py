import numpy as np
import copy
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# import time as get_time
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy import stats


# Slope acceleration
def get_slope_accelerated(location, slope_seg, slope, train_info):
    """
        Train slope acceleration
        Parameters:
         ***
        Returns:
        acc: Train slope acceleration
     """
    train_len = train_info['len']
    # Current acceleration
    slope_accelerated = 0.0
    # Gravity
    G = 9.81
    # Left-closed, right-open slope segment
    for section in range(len(slope_seg)):
        # If the train is in this slope segment (a, b]
        if slope_seg[section] < location & location <= slope_seg[section + 1]:
            # In the range from the start of this segment to the train length
            if location <= train_len + slope_seg[section]:
                # If it is in the first segment
                if section == 0:
                    slope_accelerated = (slope[section] * G * location) / (train_len * 1000)
                    break
                else:
                    slope_accelerated = (slope[section] * G * (location - slope_seg[section])) / (
                            train_len * 1000) \
                                        + (slope[section - 1] * G * (
                            train_len + slope_seg[section] - location)) / (
                                            train_len * 1000)
                    break

            # If this location exceeds the range of the segment to the train length
            if location > train_len + slope_seg[section]:
                slope_accelerated = slope[section] * G / 1000
                break

    return slope_accelerated


# Final speed per second
def get_speed(last_speed, accelerated):
    """
       Final train speed
       Parameters:
       ***
       Returns:
       speed: Final train speed
    """
    if last_speed * last_speed + 2 * accelerated >= 0:
        speed = math.sqrt(math.fabs(last_speed * last_speed + 2 * accelerated))
    else:
        speed = 0
    return speed


def get_move_time(speed, last_speed, accelerated):
    """
       Time for unit displacement of the train
       Parameters:
        ***
       Returns:
       time: Time for unit displacement
    """
    if last_speed != speed:
        time = math.fabs((speed - last_speed) / accelerated)
    else:
        time = math.fabs(1 / speed)
    return time


def get_energy(gear_accelerated, train_info):
    """
        Train energy per unit distance
        Parameters:
        ***
        Returns:
        action_energy: Train energy per unit distance
     """
    mass = train_info['mass']
    if gear_accelerated < 0:
        fa = 0
    else:
        fa = math.fabs(gear_accelerated * mass) / 3600
    action_energy = fa
    return action_energy


def get_line_step_time(line):
    ts = [0]
    t = 0
    for loc in range(1, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
        ts.append(t)
    ts = np.array(ts)
    return ts


def get_line_step_energy(line, slope_seg, slope, train_info):
    es = [0]
    e = 0
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    for loc in range(len(line) - 1):
        loc += 1  # Update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # Get slope acceleration
        last_v = line[loc - 1]  # Get last speed
        non_gear_acc = -(slope_acc + (train_info['drag_coefficient_a'] + train_info['drag_coefficient_b'] * last_v * 3.6
                         + train_info['drag_coefficient_c'] * (last_v * 3.6) ** 2) / train_info['mass'])  # Calculate acceleration
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kWh
            e += (gear_acc * mass) / 3600
        es.append(e)

    return np.array(es)


def get_seg_time(line, start, end):
    """
    Get time for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    float: Total time
    """
    t = 0
    for loc in range(start + 1, end + 1):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
    return t


def get_seg_energy(line, start, end, slope_seg, slope, train_info):
    """
    Get energy for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    float: Total energy
    """
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    c_1 = train_info['drag_coefficient_a']
    c_2 = train_info['drag_coefficient_b']
    c_3 = train_info['drag_coefficient_c']
    mass = train_info['mass']
    e = 0
    for loc in range(start, end):
        loc += 1  # Update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # Get slope acceleration
        last_v = line[loc - 1]  # Get last speed
        # Calculate basic resistance acceleration
        basic_acc = (c_1 + c_2 * last_v * 3.6 + c_3 * (last_v * 3.6) ** 2) / mass
        non_gear_acc = -(basic_acc + slope_acc)

        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kWh
            e += (gear_acc * mass) / 3600

    return e


def get_line_time(line):
    """
    Get time for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    float: Total time
    """
    t = 0
    for loc in range(1, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
    return t


def get_line_energy(line, slope_seg, slope, train_info):
    """
    Get energy for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    float: Total energy
    """
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    c_1 = train_info['drag_coefficient_a']
    c_2 = train_info['drag_coefficient_b']
    c_3 = train_info['drag_coefficient_c']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line) - 1):
        loc += 1  # Update position
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # Get slope acceleration
        last_v = line[loc - 1]  # Get last speed
        # Calculate basic resistance acceleration
        basic_acc = (c_1 + c_2 * last_v * 3.6 + c_3 * (last_v * 3.6) ** 2) / mass
        non_gear_acc = -(basic_acc + slope_acc)

        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kWh
            e += (gear_acc * mass) / 3600

    return e


def get_line_jerk(line):
    """
    Get jerk for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    float: Jerk
    """
    j = 0
    for loc in range(2, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        j += abs((line[loc] ** 2 - 2 * line[loc - 1] ** 2 + line[loc - 2] ** 2) / 2)
    return j


def discount_w(loc, line_len):
    return 1 / (1 + math.exp(-8 * ((loc / line_len) - 0.5)))


def t_error_e_loss(line, t_error, param, slope_x, slope_y):
    e = 0
    if t_error <= 0:
        return e
    else:
        delta_v = t_error / len(line)
        last_v = np.mean(np.array(line))
        mean_slope = 0
        for sp in range(len(slope_x) - 1):
            mean_slope += (slope_x[sp + 1] - slope_x[sp]) * slope_y[sp]
        mean_slope = mean_slope / slope_x[-1]
        slope_acc = mean_slope * 9.81 / 1000
        while delta_v > 0:
            train_acc = min(param['traction_140'], param['power'] / (
                    (1 + param['mass_factor']) * param['mass'] * last_v))
            basic_acc = (param['drag_coefficient_a'] + param['drag_coefficient_b'] * last_v * 3.6
                         + param['drag_coefficient_c'] * (last_v * 3.6) ** 2) / param['mass']
            now_v = math.sqrt(math.fabs(last_v * last_v + 2 * train_acc - basic_acc - slope_acc))
            if delta_v - (now_v - last_v) < 0:
                train_acc = ((last_v + delta_v) ** 2 - last_v ** 2) / 2 + basic_acc + slope_acc
            delta_v -= now_v - last_v
            last_v = now_v
            e += math.fabs(train_acc * param['mass']) / 3600
        return e


def get_v_lim_render(v_lim_x, v_lim_y):
    # Initialize x and y lists and add the starting point
    x = [v_lim_x[0]]
    y = [0]

    # Iterate over v_lim_x and v_lim_y to generate segments
    for i in range(len(v_lim_y)):
        x.append(v_lim_x[i])
        y.append(v_lim_y[i])
        if i != len(v_lim_y) - 1:
            x.append(v_lim_x[i + 1])
            y.append(v_lim_y[i])

    return np.array(x), np.array(y)


def get_slope_render(slope_x, slope_y):
    # Initialize x and y lists and add the starting point
    rel_slope_y = slope_y[0]
    x = [slope_x[0]]
    y = [slope_y[0]]

    # Iterate over slope_x and slope_y to generate segments
    for i in range(1, len(slope_y)):
        x.append(slope_x[i])
        rel_slope_y += slope_y[i - 1]
        y.append(rel_slope_y)

    return x, y


def get_render(item, gap):
    # Initialize x and y lists
    x = []
    y = []

    # Iterate over elements with the specified gap
    for i in range(0, len(item), gap):
        x.append(i)
        y.append(item[i])

    # If the last element wasn't included, add the last element
    if x[-1] != len(item) - 1:
        x.append(len(item) - 1)
        y.append(item[-1])

    return np.array(x), np.array(y)


# x, y x_max y_max conversion width conversion height shift_x shift_y
def get_line_to_pygame(x, y, x_max, y_max, width, height, shift_x=0, shift_y=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    x = shift_x + x * width / (x_max + 1)
    y = shift_y + height - y * height / (y_max + 1)
    xy = list(zip(x, y))
    return xy


def get_line_action(line, slope_seg, slope, train_info):
    """
    Get action values for each position
    Parameters:
    line (list): Speed list for the line
    Returns:
    list: Action values list
    """
    max_traction = train_info['traction_140']
    train_power = train_info['power']
    mass = train_info['mass']
    max_braking = train_info['braking_140']
    mass_factor = 1 + train_info['mass_factor']

    actions = [0]
    for loc in range(len(line) - 1):
        # Update position
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        # Get slope acceleration
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        # Get last speed
        last_v = line[loc - 1]
        basic_acc = (train_info['drag_coefficient_a'] + train_info['drag_coefficient_b'] * last_v * 3.6
                     + train_info['drag_coefficient_c'] * (last_v * 3.6) ** 2) / train_info['mass']
        # Calculate acceleration
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


def get_pmp_action(acc_pot, slope_p, slope_x, pot):
    shift = []
    front = []
    back = []

    # Use scipy.stats.mode to find the most frequent number
    crusing_v = stats.mode(pot).mode[0]

    indices = np.where(pot == crusing_v)[0]
    cruising_start, cruising_end = indices[0], indices[-1]

    for loc in range(slope_p, len(slope_x) - 1):
        if slope_x[loc + 1] >= cruising_start > slope_x[loc]:
            shift.append((cruising_start - slope_x[loc]) / (slope_x[loc + 1] - slope_x[loc]))
            front.append(1)
            back.append(max(min(acc_pot[cruising_start + 1], 1), 0))
            continue
        if slope_x[loc] < cruising_end <= slope_x[loc + 1]:
            shift.append((cruising_end - slope_x[loc]) / (slope_x[loc + 1] - slope_x[loc]))
            front.append(max(min(acc_pot[cruising_end - 1], 1), 0))
            back.append(0)
            continue
        shift.append(0.5)
        front.append(max(min(stats.mode(acc_pot[slope_x[loc]:slope_x[loc + 1]]).mode[0], 1), 0))
        back.append(max(min(stats.mode(acc_pot[slope_x[loc]:slope_x[loc + 1]]).mode[0], 1), 0))

    return [shift, front, back]
