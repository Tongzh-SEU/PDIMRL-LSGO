import numpy as np
import copy
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# import time as get_time
from scipy.interpolate import interp1d


def get_render_list(seg, info):
    list = [[], []]
    for i in range(len(seg) - 1):
        list[0].append(seg[i])
        list[1].append(info[i])
        list[0].append(seg[i + 1])
        list[1].append(info[i])
    return list


def get_train_list(list_step, n_station):
    # Initialize a new two-dimensional array to store the merged slope_step data
    # Traverse each car
    n_list = []
    list_step_ = [sublist[1:] for sublist in list_step]
    list_step_rev_ = [list(reversed(row)) for row in list_step_]

    for stops in n_station:
        # Initialize the merged slope_step data of the current vehicle
        merged_data = []
        # Traverse the stops of the current vehicle
        for i in range(len(stops) - 1):
            # Get the index of the adjacent stop
            start_station = stops[i]
            end_station = stops[i + 1]
            if end_station > start_station:
                # Merge the slope_step data of the adjacent stop
                merged_data.append(sum(list_step_[start_station: end_station], []))
                merged_data[-1].insert(0, 0)
            else:
                merged_data.append(sum(list_step_rev_[end_station: start_station], []))
                merged_data[-1].insert(0, 0)
        # Add the merged data of the current vehicle to the new two-dimensional array
        n_list.append(merged_data)
    return n_list


def get_train_list_(fast_profile, min_E_profile, n_station):
    # Initialize a new two-dimensional array to store the merged slope_step data
    # Traverse each car
    n_list_0 = []
    n_list_1 = []
    fast_profile_ = [sublist[:-1] for sublist in fast_profile]
    fast_profile_rev_ = [list(reversed(row)) for row in fast_profile]
    min_E_profile_ = [sublist[:-1] for sublist in min_E_profile]
    min_E_profile_rev_ = [list(reversed(row)) for row in min_E_profile]

    for stops in n_station:
        # Initialize the merged slope_step data of the current vehicle
        merged_data_0 = []
        merged_data_1 = []
        # Traverse the stops of the current vehicle
        for i in range(len(stops) - 1):
            # Get the index of the adjacent stops
            start_station = stops[i]
            end_station = stops[i + 1]
            if end_station > start_station:
                # Merge the slope_step data of the adjacent stops
                fast = sum(fast_profile_[start_station: end_station], [])
                min_E = sum(min_E_profile_[start_station: end_station], [])
                for j in range(1000, len(fast) - 1000):
                    if fast[j] <= 80 / 3.6:
                        fast[j] = 80 / 3.6
                    if min_E[j] <= 80 / 3.6:
                        min_E[j] = 80 / 3.6
                merged_data_0.append(fast)
                merged_data_0[-1].append(0)
                merged_data_1.append(min_E)
                merged_data_1[-1].append(0)
            else:
                fast = sum(fast_profile_rev_[end_station: start_station], [])
                min_E = sum(min_E_profile_rev_[end_station: start_station], [])
                for j in range(1000, len(fast) - 1000):
                    if fast[j] <= 80 / 3.6:
                        fast[j] = 80 / 3.6
                    if min_E[j] <= 80 / 3.6:
                        min_E[j] = 80 / 3.6
                merged_data_0.append(fast)
                merged_data_0[-1].append(0)
                merged_data_1.append(min_E)
                merged_data_1[-1].append(0)
        # Add the merged data of the current vehicle to the new two-dimensional array
        n_list_0.append(merged_data_0)
        n_list_1.append(merged_data_1)
    return n_list_0, n_list_1


def get_n_list(info_step, n_station_loc):
    n_info_step = []
    for i in range(len(n_station_loc)):
        n_info_step.append([])
        for j in range(len(n_station_loc[i]) - 1):
            n_info_step[i].append([])
            n_info_step[i][j].append(0)
            if n_station_loc[i][0] < n_station_loc[i][1]:
                n_info_step[i][j].extend(info_step[n_station_loc[i][j]:n_station_loc[i][j + 1] + 1])
            else:
                n_info_step[i][j].extend(info_step[n_station_loc[i][j + 1]:n_station_loc[i][j] + 1][::-1])
    return n_info_step


def get_list(info, start_end):
    """
    Get the slope or speed limit of each point
    Parameters:
    ***
    Return value:
    output_list: Get the slope or speed limit of each point
    """
    line_len = start_end[-1]
    output_list = [0]
    point = 0
    present_end = start_end[point + 1]
    present = info[point]

    # Open on the left and close on the right to get the slope or speed limit of each point
    for location in range(1, line_len + 1):
        if location <= present_end:
            output_list.append(present)
        elif location > present_end:
            if point == len(start_end) - 1:
                present = info[point]
                output_list.append(present)
                break
            point += 1
            present_end = start_end[point + 1]
            present = info[point]
            output_list.append(present)

    return output_list


# Get distance information for slope or speed limit
def get_remain_step(start_end):
    """
    Get the distance information of the slope or speed limit
    Parameters:
    start_end: boundary point
    Return value:
    output: distance information of the slope or speed limit
     """
    output_list = [start_end[1]]
    for grs_i in range(len(start_end) - 1):
        for remain_step in range(start_end[grs_i + 1] - start_end[grs_i] - 1, -1, -1):
            output_list.append(remain_step)
    return output_list


# Get the next slope or speed limit
def get_next_info(start_end, info, line_len):
    """
        Get the next slope/speed limit for each point
        Parameters:
        start_end: boundary point
        info: boundary point value
        Return value:
        output: Get the next slope/speed limit for each point
     """

    output = np.zeros(line_len + 1)
    point = 1
    for l in range(1, line_len + 1):
        if l > start_end[point]:
            point += 1
        output[l] = info[point]
    return output


def get_now_info(start_end, info, line_len):
    """
        Get the current slope/speed limit of each point
        Parameters:
        start_end: boundary point
        info: boundary point value
        Return value:
        output: current slope/speed limit of each point
     """
    output = np.zeros(line_len + 1)
    point = 1
    for l in range(1, line_len + 1):
        if l > start_end[point]:
            point += 1
        output[l] = info[point - 1]
    output[-1] = 0
    return output


def get_energy(gear_accelerated, train_info):
    """
        The traction capacity per unit distance of the train
        Parameters:
        ***
        Return value:
        action_energy: The traction capacity per unit distance of the train
     """
    mass = train_info['mass']
    if gear_accelerated < 0:
        fa = 0
    else:
        fa = math.fabs(gear_accelerated * mass) / 3600
    action_energy = fa
    return action_energy


def distributed_re_energy(re_energy, current_loc, other_loc, train_info):
    """
    Input
    Distribute regenerated energy
    re_energy: float Unit regenerated energy
    current_loc: float Current car location
    other_loc: dict Other cars and corresponding locations in the same power supply section key: str, value: floatt
    train_info: dict Train information
    Outputob_re_energy: dict Unit regenerated energy obtained by each car key: str, value: float

    """
    re_conv_loss_a = train_info['re_conv_loss_a']
    re_conv_loss_b = train_info['re_conv_loss_b']
    accum_len = 0
    ob_re_energy = other_loc
    for train, loc in other_loc.items():
        other_loc[train] = abs(loc - current_loc) / 1000
        accum_len += other_loc[train]
    for train, current_len in other_loc.items():
        # The regenerative energy obtained by the vehicle = regenerative energy * distribution ratio * attenuation rate
        ob_re_energy[train] = re_energy * (current_len / accum_len) * (
                    re_conv_loss_a * (current_len ** 2) + re_conv_loss_b)
    return ob_re_energy


def get_re_energy(speed, gear_accelerated, train_info):
    """
        The traction capacity per unit distance of the train
        Parameters:
        ***
        Return value:
        action_energy: The traction capacity per unit distance of the train
     """
    fa = 0
    c_70 = train_info['re_energy_a_70']
    c_294 = train_info['re_energy_b2_294']
    x_294 = train_info['re_energy_b1_294']
    c_350 = train_info['re_energy_c2_350']
    x_350 = train_info['re_energy_c1_350']
    if speed * 3.6 < 70:
        fa = c_70 / 3600
    elif speed * 3.6 < 294:
        fa = (x_294 * speed * 3.6 + c_294) / 3600
    elif speed * 3.6 < 350:
        fa = (x_350 * speed * 3.6 + c_350) / 3600

    if gear_accelerated > 0:
        fa = 0

    action_energy = fa
    return action_energy


def get_jerk(last_a, a):
    return abs(last_a - a)


'''
eg.
Slope Acceleration calculation 120 is related to the length of the train
0 at the exit
-3 0-Current slope*G*i/(train length*1000)
-Current slope*G/1000
47.83 Current slope*(i-previous slope end point)*G/(train length*1000)+previous slope*G*(previous slope end point+train distance-i)/(train length*1000)
Current slope*G/1000
-32.439 Current slope*(i-previous slope end point)*G/(train length*1000)+previous slope*(previous slope end point+train distance-i)*G/(train length*1000)
Current slope*G/1000
0 Current slope*(i-end point of previous slope)*G/(train length*1000)+previous slope*(train length-(i-end point of previous slope))*G/(train length*1000)
Current slope*G/1000
'''


def get_slope_acc(loc, P, slope_seg, slope, train_info):
    train_len = train_info['len']
    slope_accelerated = 0.0
    G = 9.81
    # Close left and open slope_seg
    if loc <= train_len + slope_seg[P]:
        if P == 0:
            slope_accelerated = (slope[P] * G * loc) / (train_len * 1000)
        else:
            slope_accelerated = (slope[P] * G * (loc - slope_seg[P])) / (train_len * 1000) + (slope[P - 1] * G *
                                                                                              (train_len + slope_seg[
                                                                                                  P] - loc)) / (
                                            train_len * 1000)
            # This position exceeds the length of the interval to the train
    if loc > train_len + slope_seg[P]:
        slope_accelerated = slope[P] * G / 1000

    return slope_accelerated


# Slope acceleration
def get_slope_accelerated(location, slope_seg, slope, train_info):
    """
        Train slope acceleration
        Parameters:
        ***
        Return value:
        acc: Train slope acceleration
     """
    train_len = train_info['len']
    # Current acceleration
    slope_accelerated = 0.0
    G = 9.81
    # Left closed right open slope_seg
    for section in range(len(slope_seg)):
        # If he is in this slope interval (a,b]
        if slope_seg[section] < location & location <= slope_seg[section + 1]:
            # From the beginning of this section to the train length
            if location <= train_len + slope_seg[section]:
                # If he is in the first section
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

            # This position exceeds the section to the train length position
            if location > train_len + slope_seg[section]:
                slope_accelerated = slope[section] * G / 1000
                break

    return slope_accelerated


def get_basic_acc(loc, last_v, train_info):
    """
        The actual acceleration of the train
        Parameters:
        last_speed: final speed
        gear_accelerated: train traction acceleration：
        slope_accelerated: slope accelerationd：
        loc: location
        train_info dictionary
        ***
        Return value:
        acc: the actual acceleration of the train
     """
    # return gear_accelerated - slope_accelerated - ( .drag_coefficient_a + .drag_coefficient_b * last_speed * 3.6 + .drag_coefficient_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    mass = train_info['mass']

    G = 9.81
    if loc == 1:
        return 0
    else:
        basic_acc = (co_a + co_b * last_v * 3.6 + co_c * (last_v * 3.6) ** 2) / mass
        return basic_acc


# Comprehensive acceleration = gear acceleration - slope acceleration - basic resistance acceleration
def get_accelerated(last_v, gear_acc, slope_acc, loc, train_info):
    """
    The actual acceleration of the train
    Parameters:
    last_speed: final speeded：
    gear_accelerated: train traction accelerationelerated：
    slope_accelerated: slope accelerationcelerated：
    loc: location
    train_info dictionaryfo
    ***
    Return value:
    acc: the actual acceleration of the train
    """
    # return gear_accelerated - slope_accelerated - ( .drag_coefficient_a + .drag_coefficient_b * last_speed * 3.6 + .drag_coefficient_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    mass = train_info['mass']

    # 重力
    G = 9.81
    if loc == 1:
        return gear_acc - slope_acc
    else:
        basic_acc = (co_a + co_b * last_v * 3.6 + co_c * (last_v * 3.6) ** 2) / mass
        return gear_acc - slope_acc - basic_acc


def get_basic_accelerated(last_speed, loc, train_info):
    """
    Basic resistance acceleration of the train
    Parameters:
    ****
    Return value::
    acc: Basic resistance acceleration of the train:
    """
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_b']
    G = 9.81
    if loc == 1:
        return 0
    else:
        return (
                co_a + co_b * last_speed * 3.6 + co_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000


# 每一秒的末速度
def get_speed(last_speed, accelerated, loc):
    """
       Final speed of train operation
       Parameter:
       ***
       Return value:
       speed: Final speed of train operation
    """
    if loc == 1:
        return math.sqrt(math.fabs(last_speed * last_speed + 2 * accelerated))
    else:
        if last_speed * last_speed + 2 * accelerated >= 0:
            speed = math.sqrt(math.fabs(last_speed * last_speed + 2 * accelerated))
        else:
            speed = 0
    return speed



def get_move_time(speed, last_speed, accelerated):
    """
       Unit displacement time of train
       Parameter:
       ***
       Return value:
       time: Unit displacement time of train
    """
    if last_speed != speed:
        time = math.fabs((speed - last_speed) / accelerated)
    else:
        time = math.fabs(1 / speed)
    return time


def get_py_draw_line(x, y, x_rate=0.5, y_rate=0.5, x_shift=0.25, y_shift=0.25, line_len=40000):
    screen_width = 700
    screen_height = 650
    x_zoom = screen_width / (line_len + 1)
    y_zoom = screen_height / 310

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    speed_limit_xs = x * x_zoom
    speed_limit_ys = y * y_zoom
    speed_limit_xys = \
        list(zip((speed_limit_xs * x_rate + screen_width * x_shift),
                 (speed_limit_ys * y_rate + screen_height * y_shift)))
    return speed_limit_xys


# x, y x_max y_max Convert width Convert length Offset x Offset y
def get_line_to_pygame(x, y, x_max, y_max, width, height, shift_x=0, shift_y=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    x = shift_x + x * width / (x_max + 1)
    y = shift_y + height - y * height / (y_max + 1)
    xy = list(zip(x, y))
    return xy


# Original coordinates x, y, original area size x_max, y_max,
# newly generated area size width, height, offset relative to
# the main canvas shift_x, shift_y
def get_point_to_pygame(x, y, x_max, y_max, width, height, shift_x=0, shift_y=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    x = shift_x + x * width / (x_max + 1)
    y = shift_y + height - y * height / (y_max + 1)
    return x, y


def get_height(slope, slope_seg):
    """
       gym canvas height
       Return value:
       height_y: canvas height
    """

    # Initialize the height list, setting the initial height to 0
    height = [0]
    # Iterate through the slope list, calculate the height corresponding to each slope, and append it to the height list
    for slope_i in range(len(slope) - 1):
        height.append(round(slope[slope_i] + height[slope_i], 3))
    # Initialize the height_y list, setting the initial height to 0
    height_y = [0]
    # Iterate through the height list, calculate the y-coordinate for each height segment, and append it to the height_y list
    for height_i in range(len(height) - 1):
        # Calculate the number of intervals for the current height segment
        space = int(float(slope_seg[height_i + 1] - slope_seg[height_i]))
        # Get the starting height and ending height of the current segment
        start = float(height[height_i])
        end = float(height[height_i + 1])
        # Use numpy's linspace function to generate equally spaced height values between the start and end height
        temp = np.linspace(start, end, space)
        # Concatenate the generated height values to the height_y list
        height_y = np.concatenate((height_y, temp))
    # Return the height_y list
    return height_y


def get_length(point):
    """
       Get segment length
       Parameters:
       point: boundary point
       Return value:
       length: segment length
    """
    length = []
    for p_i in range(len(point) - 1):
        length.append(point[p_i + 1] - point[p_i])
    return length


def get_avg_v(len_seg, time_seg):
    """
       Get average speed
       Parameters:
       len_seg: segment length
       time_seg: segment time
       Return value:
       avg_v: segment average speed
    """
    avg_v = []
    for len_i, time_i in zip(len_seg, time_seg):  # Iterate over the length and time of each segment
        avg_v.append(len_i / time_i)  # Calculate the average speed and add to the list
    return avg_v


def get_v_sort(avg_v):
    """
       Get the sorting speed of the segment
       Parameters:
       avg_v: average speed of the segment
       Return value:
       seg: segment: index
       v_sort: sorted segment speed
    """
    seg = np.arange(0, len(avg_v), 1)  # Initialize the seg array to contain integers from 0 to len(avg_v)-1
    v_sort = copy.deepcopy(avg_v)  # Copy avg_v to v_sort

    for i in range(len(seg)):
        for j in range(0, len(seg) - i - 1):
            if v_sort[j] < v_sort[j + 1]:  # If the previous element is smaller than the next element
                v_sort[j], v_sort[j + 1] = v_sort[j + 1], v_sort[j]
                seg[j], seg[j + 1] = seg[j + 1], seg[j]  # Also swap the positions of corresponding elements in the seg array

    for i in range(len(v_sort) - 1):
        if v_sort[i] != v_sort[i + 1] and math.fabs(v_sort[i + 1] - v_sort[i]) <= 0.0000000001:
            v_sort[i + 1] = v_sort[i]
    return seg, v_sort


def get_block(seg, v_sort):
    """
       Get the total length of the block
       Parameters:
       seg: segment
       v_sort: speed sort from large to small
       Return value:
       block: block: a set of segments with the same average speed
    """

    groups = defaultdict(list)
    for i in range(len(v_sort)):
        groups[v_sort[i]].append(seg[i])
    block = []
    for key in sorted(groups.keys(), reverse=True):
        block.append(groups[key])
    return block


def get_block_sum_len(block, seg_len):
    """
       Get the total length of the block.
       Parameters:
        seg_len: A collection of segment lengths.
        block: A block; a set of segments with the same average speed.
       Returns:
       block_sum_len: The total length of the block.
    """
    block_sum_len = []
    for group in block:  # Iterate over each block in the collection
        group_sum = 0
        for bsl_index in group:  # Iterate over each segment in the group
            group_sum += seg_len[bsl_index]  # Add the corresponding element from seg_len to the group sum
        block_sum_len.append(group_sum)  # Append the group sum to block_sum_len
    return block_sum_len


def get_block_len(block, seg_len):
    """
       Get the length collection of the block.
       Parameters:
        seg_len: A collection of segment lengths.
        block: A block; a set of segments with the same average speed.
       Returns:
       block_len: The collection of block lengths.
    """
    block_len = []
    for group in block:  # Iterate over each block in the collection
        group_values = []
        for bl_index in group:  # Iterate over each segment in the group
            group_values.append(seg_len[bl_index])  # Append the corresponding element from seg_len to the group values
        block_len.append(group_values)  # Append the group values to block_len
    return block_len


def get_block_avg_v(block, avg_v):
    """
       Get the collection of average speeds for the block.
       Parameters:
        avg_v: A collection of segment average speeds.
        block: A block; a set of segments with the same average speed.
       Returns:
       block_avg_v: The collection of average speeds for the block.
    """
    block_avg_v = []
    for i in range(len(block)):  # Iterate over each block in the collection
        block_avg_v.append(avg_v[block[i][0]])  # Get the average speed for each block
    return block_avg_v


def allocate_runtime(seg_time, block, block_len, block_sum_len, rs_time):
    """
       Calculate the psi value and maximum rate ratio based on MRI data and other parameters.
       Parameters:
        seg_time: The running time of each segment.
        block: A block; a set of segments with the same average speed.
        block_len: The length of each block.
        block_sum_len: The total length of all blocks.
        rs_time: The time allocated to this block.
       Returns:
       seg_time: The updated running time for the segments.
    """
    first_block = block[0]
    first_block_len = block_len[0]
    first_block_sum_len = block_sum_len[0]
    for i in range(len(first_block)):  # Iterate over each element in the first group
        seg_index = first_block[i]  # Get the element's position in seg_time
        seg_length = first_block_len[i]  # Get the corresponding segment length
        allocated_time = (seg_length / first_block_sum_len) * rs_time  # Calculate the time allocated to this element
        seg_time[seg_index] += allocated_time  # Add the allocated time to the corresponding element in seg_time
    return seg_time



def get_area_list(seg, info, station):
    area_seg_speed_lim = []
    area_speed_lim = []

    for s in sorted(station):
        if s not in seg:
            index = next(x[0] for x in enumerate(seg) if x[1] > s)
            seg.insert(index, s)
            info.insert(index, info[index - 1])

    index = 0

    for i in range(len(station) - 1):
        if station[0] < station[1]:
            start = station[i]
            end = station[i + 1]
        else:
            start = station[i]
            end = station[i + 1]
        area_seg = []
        area = []
        while True:
            area_seg.append(seg[index] - start)
            area.append(info[index])
            index += 1
            if seg[index] == end:
                area_seg.append(end - start)
                area.append(0)
                area_seg_speed_lim.append(area_seg)
                area_speed_lim.append(area)
                break

    return area_seg_speed_lim, area_speed_lim


def clip_to_n_station(seg, info, station_loc):
    n_station_seg = []
    n_station_info = []
    station_P = 1
    shift_i = 0
    shift_loc = 0
    for i in range(len(seg) - 1):
        if seg[i] < station_loc[station_P] <= seg[i + 1]:
            n_station_seg.append(np.array(seg[shift_i: i + 1]) - shift_loc)
            n_station_info.append(np.array(info[shift_i: i + 1]))
            if station_P != 1:
                n_station_seg[station_P - 1] = np.insert(n_station_seg[station_P - 1], 0, 0)
                n_station_info[station_P - 1] = np.insert(n_station_info[station_P - 1], 0,
                                                          n_station_info[station_P - 2][-2])

            n_station_seg[station_P - 1] = np.append(n_station_seg[station_P - 1], station_loc[station_P] - shift_loc)
            n_station_info[station_P - 1] = np.append(n_station_info[station_P - 1], 0)
            shift_i = i + 1
            shift_loc = station_loc[station_P]
            station_P += 1
    return n_station_seg, n_station_info


def get_mri(speed_lim_seg, speed_lim, slope_seg, slope, train_info):
    """
    Shortest running time trajectory
    speed_lim_seg unit m
    speed_lim unit km/h
    train_info is a dictionary
    Return value:
    np.array(mri): Shortest running time trajectory curve
    seg_t: Speed of each speed limit segment
    v_frontier: Traction-cruise mode transition point
    """
    # Initialize v[0] as the acceleration speed from the starting point, v[1] as the deceleration speed from the end point
    # v_frontier is the position or speed limit point where the speed limit is reached
    # v_lim is the speed limit per meter

    line_len = speed_lim_seg[-1] - speed_lim_seg[0]

    v = np.array([np.zeros(line_len + 1), np.zeros(line_len + 1)])
    v_frontier = np.array([speed_lim_seg, speed_lim_seg])
    v_lim = get_now_info(speed_lim_seg, speed_lim, line_len)
    max_traction = train_info['traction_140']
    power = train_info['power']
    mass_factor = train_info['mass_factor']
    mass = train_info['mass']
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    max_braking = train_info['braking_140']
    train_len = train_info['len']
    seg = 1
    loc = 0
    # Record acceleration from left to right
    while True:
        loc += 1
        # Get the acceleration
        if v[0][loc - 1] == 0:
            max_gear_action = max_traction
        else:
            max_gear_action = min(max_traction, power / (
                    (1 + mass_factor) * mass * v[0][loc - 1]))

        gear_acc = max_gear_action
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = v[0][loc - 1]
        acc = get_accelerated(last_v, gear_acc, slope_acc, loc, train_info)
        # Update speed, ensuring it stays below the speed limit
        now_v = min(v_lim[loc] / 3.6, math.sqrt(last_v ** 2 + 2 * acc))
        # Store the speed
        v[0][loc] = now_v
        # If the speed limit is reached, move to the next speed limit starting point
        if now_v == v_lim[loc] / 3.6:
            for i in range(len(speed_lim_seg)):
                if speed_lim_seg[i] < loc <= speed_lim_seg[i + 1]:
                    v_frontier[0][seg] = loc
                    loc = speed_lim_seg[i + 1]
                    if loc != speed_lim_seg[-2]:
                        v[0][loc] = min(now_v, v_lim[loc + 1] / 3.6)
                    seg += 1
                    break
        # If the speed limit is not reached but the speed limit point is reached
        if loc == speed_lim_seg[seg]:
            seg += 1
        # End point
        if loc == speed_lim_seg[-2]:
            break

    seg = -2
    loc = line_len + 1

    # Record deceleration from right to left
    while True:
        loc -= 1
        # Get the deceleration
        gear_acc = max_braking
        slope_acc = get_slope_accelerated(loc - 1, slope_seg, slope, train_info)
        # now_v**2 - last_v**2 = 2(gear_acc - slope_acc - basic_acceleration) * train_step
        # basic_acceleration = a + b*last_v*3.6 + c*(last_v*3.6)**2
        # Solve for last_v
        now_v = v[1][loc]
        if now_v != v_lim[loc - 1] / 3.6:
            # Solve the quadratic equation. Since a is always less than 0, there is only one valid solution (v > 0)
            a = 1 - 2 * co_c * 3.6 ** 2 * (9.81 / 1000)
            b = -2 * co_b * 3.6 * (9.81 / 1000)
            c = -now_v ** 2 + 2 * (gear_acc - slope_acc - co_a * (9.81 / 1000))
            last_v = min(v_lim[loc - 1] / 3.6, (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        else:
            last_v = v_lim[loc - 1] / 3.6
        # Store the speed
        v[1][loc - 1] = last_v
        # If the speed limit is reached, move to the next speed limit starting point
        if last_v == v_lim[loc] / 3.6:
            for i in range(len(speed_lim_seg)):
                if speed_lim_seg[i] < loc <= speed_lim_seg[i + 1]:
                    v_frontier[1][seg] = loc
                    loc = speed_lim_seg[i] + 1
                    if loc != speed_lim_seg[1] + 1:
                        v[1][loc - 1] = last_v
                    seg -= 1
                    break
        # If the speed limit is not reached but the speed limit point is reached
        if loc == speed_lim_seg[seg]:
            seg -= 1
        # End point
        if loc == speed_lim_seg[1] + 1:
            break

    for v_i in range(len(v_lim)):
        if v_i != 0 and v_i != len(v_lim) - 1:
            if v[0][v_i] == 0:
                v[0][v_i] = v_lim[v_i] / 3.6
            if v[1][v_i] == 0:
                v[1][v_i] = v_lim[v_i] / 3.6
    mri = np.minimum(v[0], v[1])

    # Calculate the shortest running time for each segment
    i = 0
    seg_t = np.zeros(len(speed_lim_seg) - 1)
    for loc in range(len(mri)):
        loc += 1
        seg_t[i] += 2 / (mri[loc - 1] + mri[loc])
        if loc == speed_lim_seg[i + 1]:
            i += 1
        if mri[loc] == 0:
            break

    return np.array(mri), sum(seg_t), seg_t, v_frontier






def get_psi_part_a(mri, v_frontier, plan_time):
    """
    Calculate psi value and maximum rate ratio based on MRI data and other parameters
    Parameters:
    psi_a (float): original psi value
    mri (float): mri value
    v_frontier (float): traction-cruise boundary point
    sat_seg_t (float): sat segment time
    rate (float): conversion rate ratio
    Return value:
    psi: 1D array representing the calculated psi value
    max_rate: maximum rate ratio
    """

    psi = np.array(mri)
    is_done = False

    low = 0
    high = v_frontier[0][2]

    while True:
        l_cr = (low + high) // 2
        v_cr = np.ones(len(psi)) * psi[l_cr]
        temp_psi = np.minimum(v_cr, psi)
        temp_time = get_line_time(temp_psi)
        if temp_time < plan_time:
            high = l_cr - 1
        else:
            low = l_cr + 1

        if low >= high:
            psi = copy.deepcopy(temp_psi)
            break



    max_rate = np.max(np.array(mri)) / np.max(psi)
    return psi, max_rate, psi[l_cr]


def get_psi_part_b(psi, mri, v_frontier, plan_time, rate, line_len, speed_lim_seg, slope_seg, slope, train_info,
                   last_cov=0, last_rate=0):

    psi_rate = np.zeros(line_len + 1)

    for i, (psi_value, mri_value) in enumerate(zip(psi, mri)):
        if psi_value * rate >= mri_value:
            psi_rate[i] = mri_value
        else:
            psi_rate[i] = psi_value * rate
    if last_cov == 0:
        low = v_frontier[0][2]
        high = line_len
    else:
        if abs(last_rate - rate) <= 0.2:
            low = max(int(last_cov - line_len * 0.1), v_frontier[0][2])
            high = min(int(last_cov + line_len * 0.1), line_len)
        else:
            low = v_frontier[0][2]
            high = line_len
    convert = [0, 0]
    while True:
        coasting_loc = (low + high) // 2
        temp_psi = copy.deepcopy(psi_rate)
        is_renew = False
        for loc in range(coasting_loc, line_len):
            loc += 1
            gear_action = 0
            slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 调用get_slope_accelerated方法获取slope_acc
            last_v = temp_psi[loc - 1]
            acc = get_accelerated(last_v, gear_action, slope_acc, loc, train_info)  # 调用get_accelerated方法获取加速度
            if last_v ** 2 + 2 * acc < 0:
                now_v = 0
            else:
                now_v = math.sqrt(last_v ** 2 + 2 * acc)
            if now_v == 0 and loc != line_len:
                is_renew = True
                break
            temp_psi[loc] = now_v
            if temp_psi[loc] >= psi_rate[loc]:
                temp_psi[loc:] = psi_rate[loc:]
                if loc > speed_lim_seg[-2]:
                    break
                '''
                else:
                    loc = speed_lim_seg[-2]
                    continue
                '''
        if is_renew:
            low = coasting_loc + 1
            continue
        t = get_line_time(temp_psi)

        convert = [coasting_loc, temp_psi[coasting_loc]]  # 更新convert数组的值
        if convert[0] == line_len - 1:
            convert = [0, 0]
        if t > plan_time:
            low = coasting_loc + 1
        else:
            high = coasting_loc - 1
        if low >= high and line_len <= 40000:
            psi_rate = copy.deepcopy(temp_psi)
            break
        if low + 10 >= high and line_len > 40000:
            psi_rate = copy.deepcopy(temp_psi)
            break
    return psi_rate, convert


def get_line_step_time(line):
    ts = [0]
    t = 0
    for loc in range(1, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
        ts.append(t)
    return ts


def get_line_step_energy(line, slope_seg, slope, train_info):
    es = [0]
    e = 0
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    for loc in range(len(line) - 1):
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)  # 计算加速度
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kwh
            e += (gear_acc * mass) / 3600
        es.append(e)

    return es


def get_line_time(line):

    t = 0
    for loc in range(1, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
    return t


def get_line_cr_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line) - 1):
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度
        last_v = line[loc - 1]
        v = line[loc]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)  # 计算加速度
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0 and last_v == v:
            # kwh
            e += (gear_acc * mass) / 3600

    return e


def get_line_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line) - 1):
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)  # 计算加速度
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kwh
            e += (gear_acc * mass) / 3600

    return e


def get_line_re_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line) - 1):
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)  # 计算加速度
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc < 0:
            # kwh
            e += (abs(gear_acc) * mass) / 3600

    return e


def get_line_action(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    train_power = train_info['power']
    mass = train_info['mass']
    max_braking = train_info['braking_140']

    actions = [0]
    for loc in range(len(line) - 1):

        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2

        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度

        last_v = line[loc - 1]

        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)

        if line[loc - 1] == 0:
            max_gear_action = max_traction
        else:
            max_gear_action = min(max_traction, train_power / (mass * line[loc - 1]))

        gear_acc = max(min(out_acc - non_gear_acc, max_gear_action), max_braking)
        if gear_acc >= 0:
            action = gear_acc / max_gear_action
        else:
            action = gear_acc / max_braking
        actions.append(action)
    return actions


def get_psi_min_energy(psi_a, mri, v_frontier, max_rate, plan_time, slope_seg, slope, train_info, line_len,
                       speed_lim_seg):
    """
    Get the psi_b and speed ratio corresponding to the minimum energy
    Parameters:
    psi_a (float): original psi value
    mri (float): mri value
    v_frontier (float): traction-cruise boundary point
    sat_seg_t (float): sat segment time
    max_rate (float): maximum speed ratio
    Return:
    tuple: minimum energy and corresponding speed ratio
    """

    rates = np.linspace(1, max_rate, 100)

    start = 0
    end = len(rates) - 1

    best_energy = get_line_energy(psi_a, slope_seg, slope, train_info)
    best_energy_rate = rates[0]


    psi_b, cov = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rates[(start + end) // 2], line_len, speed_lim_seg,
                                slope_seg, slope, train_info)
    energy = get_line_energy(psi_b, slope_seg, slope, train_info)

    t1 = get_line_time(psi_a)
    t2 = get_line_time(psi_b)


    while start < end:
        mid = (start + end) // 2

        time = math.fabs(plan_time - get_line_time(psi_b))


        psi_left, cov_left = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rates[(start + mid) // 2], line_len,
                                            speed_lim_seg, slope_seg, slope, train_info)
        energy_left = get_line_energy(psi_left, slope_seg, slope, train_info)
        time_left = math.fabs(plan_time - get_line_time(psi_left))

        # print('left_rate:', round(rates[(start + mid) // 2], 4))
        # print('left_cov:', cov_left[0], ' cov_rate = ', round(cov_left[0]/line_len,4))

        #
        psi_right, cov_right = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rates[(end + mid) // 2], line_len,
                                              speed_lim_seg, slope_seg, slope, train_info)
        energy_right = get_line_energy(psi_right, slope_seg, slope, train_info)
        time_right = math.fabs(plan_time - get_line_time(psi_right))

        # print('right_rate:', round(rates[(end + mid) // 2], 4))
        # print('right_cov:', cov_right[0], ' cov_rate = ', round(cov_left[0]/line_len,4))

        #
        if energy <= best_energy:
            best_energy = energy
            best_energy_rate = rates[mid]

        if abs(energy_right - energy_left) <= 0.5:
            break

        #
        # if max(time, time_left, time_right) >= 1 or energy_right >= energy_left:
        if energy_right >= energy_left:
            end = mid
            psi_b = psi_left
            energy = energy_left
            continue
        if energy_left > energy_right:
            start = mid
            psi_b = psi_right
            energy = energy_right

    return best_energy, best_energy_rate


def get_on_time_max_rate(psi_a, mri, v_frontier, max_rate, plan_time, line_len, speed_lim_seg, slope_seg, slope,
                         train_info):

    #
    rates = np.linspace(1, max_rate, 100)
    left = 0
    right = len(rates) - 1

    #
    while left < right:
        mid = (left + right) // 2
        psi_b, cnv = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rates[mid], line_len, speed_lim_seg, slope_seg,
                                    slope, train_info)
        time_error = math.fabs(plan_time - get_line_time(psi_b))
        if time_error >= 1:
            right = mid
        else:
            left = mid + 1

    boundary = left
    return rates[boundary]


#
def get_bessel_curve(psi_a, mri, v_frontier, plan_time, max_rate, energy_saving_rate, line_len, speed_lim_seg,
                     slope_seg, slope, train_info, precision=100):

    rates = np.logspace(np.log10(1), np.log10(max_rate), num=10)

    x = []
    y = []
    max_psi_b = psi_a
    min_psi_b = psi_a

    #
    for rate in rates:
        psi_b, cnv = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rate, line_len, speed_lim_seg, slope_seg, slope,
                                    train_info)

        max_psi_b = np.maximum(max_psi_b, psi_b)
        min_psi_b = np.minimum(min_psi_b, psi_b)
        x.append(cnv[0])
        y.append(cnv[1])
    x = np.array(x)
    y = np.array(y)

    #
    i = 0
    while True:
        if x[i] == x[i + 1] or x[i] == 0:
            x = np.delete(x, i)
            y = np.delete(y, i)
            i -= 1
        i += 1
        if i == len(x) - 1:
            break

    #
    try:
        f = interp1d(x, y, kind='cubic')
    except:
        f = interp1d(x, y, kind='linear')
    # f = interp1d(x, y, kind='cubic')
    x = np.arange(np.min(x), np.max(x), 1)
    y = f(x)
    bessel = np.zeros(line_len + 1)
    bessel[x] = y

    return np.maximum(max_psi_b, bessel), min_psi_b


def get_actions(psi_a, mri, v_frontier, plan_time, max_rate, line_len, speed_lim_seg, slope_seg, slope, train_info,
                precision=100):
    rates = np.linspace(1, max_rate, precision)
    actions = []
    for rate in rates:
        psi_b, _ = get_psi_part_b(psi_a, mri, v_frontier, plan_time, rate, line_len, speed_lim_seg, slope_seg, slope,
                                  train_info)
        actions.append(get_line_action(psi_b, slope_seg, slope, train_info))
    actions = np.array(actions)
    return actions


'''
# 
def plot_line(psi_a, mri, v_frontier, sat_seg_t, max_rate, line_len, num=255):
    rates = np.linspace(1, max_rate, num)  # 在1到max_rate之间生成num个等间距的数
    colors = np.linspace(0x11, 0xff, num)  # 在0x11到0xff之间生成num个等间距的数
    for i in range(num):
        psi_b, cnv_b = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rates[i])  # 调用get_psi_part_b函数获取psi_b和cnv_b的值
        index = np.arange(0, line_len + 1, 1)  # 在0到pram.line_len + 1之间生成步长为1的数列
        plt.plot(index, psi_b * 3.6, color='#50ef' + str(hex(int(colors[i])))[2:])  # 用生成的数值绘制线条，颜色使用十六进制表示
'''


def anticipate_energy_target(mri, psi, v_cr, max_v_lim, seg_slope, slope, train_info):
    basic_acc = get_basic_acc(0, v_cr, train_info)
    mt_acc = min(train_info['traction_140'], train_info['power'] / (
            (1 + train_info['mass_factor']) * train_info['mass'] * v_cr))
    cr_acc = basic_acc
    mt_energy = get_energy(mt_acc, train_info)
    cr_energy = get_energy(cr_acc, train_info)
    fast_cr_energy = get_line_cr_energy(mri, seg_slope, slope, train_info)
    min_cr_energy = get_line_cr_energy(psi, seg_slope, slope, train_info)

    v_lim_negative = 2 * (1 / (1 + np.exp(1) ** (-0.1 * (max_v_lim - v_cr * 3.6))) - 1)
    traction_cost_positive = 1 - (cr_energy / mt_energy)
    cruise_mileage_positive = min_cr_energy / fast_cr_energy

    anticipate_energy = min_cr_energy * 0.1 * (v_lim_negative + (traction_cost_positive + cruise_mileage_positive) / 2)


    return anticipate_energy


def allocation_surplus_time(min_time, seg_times, speed_lim_seg, plan_time):
    point = speed_lim_seg
    # plan time
    p_time = plan_time
    # segment length
    seg_lens = get_length(point)
    # total supplement time
    ts_time = p_time - min_time
    if ts_time < 0:
        raise Exception(f"Planned time {p_time:.2f} is less than the minimum running time {min_time:.2f}", )

    print(f"{min_time:.2f},{p_time:.2f},{ts_time:.2f}")

    while True:
        # Average speed
        avg_vs = get_avg_v(seg_lens, seg_times)
        # Rank segments by speed, segment sorting seg = [sge1_loc,...,segN_loc], v_sort = [avg_v1,...avg_vN] from Max to Min
        seg, v_sort = get_v_sort(avg_vs)
        # Speed blocks, block = [block1[seg1,...,segN]...Max to Min, blockN]
        block = get_block(seg, v_sort)
        # Get block lengths
        block_sum_len = get_block_sum_len(block, seg_lens)
        block_avg_v = get_block_avg_v(block, avg_vs)
        block_len = get_block_len(block, seg_lens)

        # Calculate required supplementary time
        delta_rs_time = 1 / block_avg_v[1] - 1 / block_avg_v[0]
        rs_time = delta_rs_time * float(block_sum_len[0])
        # If the required time exceeds the available time, allocate all remaining time to rs
        if ts_time < rs_time:
            rs_time = ts_time
        ts_time -= rs_time

        # Allocate runtime across segments
        seg_times = allocate_runtime(seg_times, block, block_len, block_sum_len, rs_time)
        avg_vs = get_avg_v(seg_lens, seg_times)
        if ts_time == 0:
            break

    return seg_times, avg_vs


def planing_speed_interval_mini(mri, v_frontier, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg, v_lim=310):
    # Find the Traction-Cruise-Traction curve
    psi_a, max_rate, v_cr = get_psi_part_a(mri, v_frontier, plan_time)
    # Find the minimum energy value and the minimum energy rate
    min_energy, min_energy_rate = get_psi_min_energy(psi_a, mri, v_frontier, max_rate, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg)
    print("Maximum energy:", round(get_line_energy(mri, slope_seg, slope, train_info), 2))
    print("Threshold energy consumption:", round(min_energy, 2))
    # Obtain the minimum energy curve: Traction-Cruise-Coasting-Traction
    min_energy_psi, _ = get_psi_part_b(psi_a, mri, v_frontier, plan_time, min_energy_rate, line_len, speed_lim_seg, slope_seg, slope, train_info)
    aet_energy = anticipate_energy_target(mri, min_energy_psi, v_cr, v_lim, slope_seg, slope, train_info)
    # Get the actions of the minimum energy curve
    min_energy_action = get_line_action(min_energy_psi, slope_seg, slope, train_info)
    min_energy_times = get_line_step_time(min_energy_psi)
    min_energy_step = get_line_step_energy(min_energy_psi, slope_seg, slope, train_info)

    return min_energy_psi, min_energy, aet_energy, min_energy_action, min_energy_times, min_energy_step


def planing_speed_interval(mri, v_frontier, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg, v_lim=310):
    # Find the Traction-Cruise-Traction curve
    psi_a, max_rate, v_cr = get_psi_part_a(mri, v_frontier, plan_time)
    # Find the minimum energy value and the minimum energy rate
    min_energy, min_energy_rate = get_psi_min_energy(psi_a, mri, v_frontier, max_rate, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg)
    print("Maximum energy:", round(get_line_energy(mri, slope_seg, slope, train_info), 2))
    print("Threshold energy consumption:", round(min_energy, 2))
    # Obtain the minimum energy curve: Traction-Cruise-Coasting-Traction
    min_energy_psi, _ = get_psi_part_b(psi_a, mri, v_frontier, plan_time, min_energy_rate, line_len, speed_lim_seg, slope_seg, slope, train_info)
    aet_energy = anticipate_energy_target(mri, min_energy_psi, v_cr, v_lim, slope_seg, slope, train_info)
    # Get the actions of the minimum energy curve
    min_energy_action = get_line_action(min_energy_psi, slope_seg, slope, train_info)
    min_energy_times = get_line_step_time(min_energy_psi)
    min_energy_step = get_line_step_energy(min_energy_psi, slope_seg, slope, train_info)
    # Find the maximum rate that meets the time requirement
    on_time_max_mate = get_on_time_max_rate(psi_a, mri, v_frontier, max_rate, plan_time, line_len, speed_lim_seg, slope_seg, slope, train_info)
    # Find the maximum rate that is energy-saving and meets the time requirement
    max_rate = min(max(min_energy_rate + (min_energy_rate - 1), min_energy_rate * 1.1), max_rate)

    # Calculate the upper and lower bounds for psi_a, mri, v_frontier, sat_seg_t, max_rate, energy_saving_rate
    upper_bound_psi, lower_bound_psi = get_bessel_curve(psi_a, mri, v_frontier, plan_time, max_rate, min_energy_rate, line_len, speed_lim_seg, slope_seg, slope, train_info)

    return min_energy_psi, upper_bound_psi, lower_bound_psi, min_energy, aet_energy, min_energy_action, min_energy_times, min_energy_step
