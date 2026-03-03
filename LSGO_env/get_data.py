import numpy as np
import pandas as pd
import data_utils as utils
import copy
import time
import os
import pickle
import matplotlib.pyplot as plt

# Main entry point for the script
if __name__ == '__main__':
    # Load slope, speed limit, and train information from CSV files
    df_slope = pd.read_csv('data/parameter/slope.csv')
    df_speed_lim = pd.read_csv('data/parameter/speed_limit.csv')
    df_train = pd.read_csv('data/parameter/train.csv')

    # Convert train data to a list of dictionaries for easy access
    train_info = df_train.apply(lambda row: row.astype(float).to_dict(), axis=1).to_list()[0]

    # Determine the number of stations based on the slope data
    station_num = len(df_slope.groupby('station'))
    station = [i for i in range(station_num)]

    # Group and extract segment slopes and speed limits for each station
    n_seg_slope = dict(
        zip(station, (df_slope.groupby('station')['seg_slope'].apply(list).reset_index())['seg_slope'].tolist()))
    n_slope = dict(zip(station, (df_slope.groupby('station')['slope'].apply(list).reset_index())['slope'].tolist()))
    n_seg_v_lim = dict(
        zip(station, (df_speed_lim.groupby('station')['seg_v_lim'].apply(list).reset_index())['seg_v_lim'].tolist()))
    n_v_lim = dict(zip(station, (df_speed_lim.groupby('station')['v_lim'].apply(list).reset_index())['v_lim'].tolist()))

    # Store the length of each station (last segment in each station)
    n_station_len = {s: seg[-1] for s, seg in n_seg_v_lim.items()}

    # Initialize dictionaries for storing speed profiles, energy parameters, etc.
    n_mri = dict(zip(station, [[] for _ in range(station_num)]))
    n_speed_ub = dict(zip(station, [[] for _ in range(station_num)]))
    n_speed_lb = dict(zip(station, [[] for _ in range(station_num)]))
    n_speed_pmp = dict(zip(station, [[] for _ in range(station_num)]))
    low_speed = dict(zip(station, [[] for _ in range(station_num)]))
    high_speed = dict(zip(station, [[] for _ in range(station_num)]))
    speed_index = dict(zip(station, [[] for _ in range(station_num)]))

    # Initialize energy-related dictionaries with default values
    n_energy_eff_save = dict(zip(station, [[0.05 for _ in range(10)] for i in range(station_num)]))
    n_energy_thr = dict(zip(station, [[1500 for _ in range(10)] for i in range(station_num)]))
    n_energy_weight = dict(zip(station,
                               [[3 * (thr / save) for save, thr in zip(n_energy_eff_save[i],
                                                                       n_energy_thr[i])] for i
                                in range(station_num)]))
    n_minE_action = dict(zip(station, [[] for _ in range(station_num)]))

    n_p_time = []
    n_p_index = []

    # Check if time.csv file exists, if not, calculate and save planned times
    if not os.path.exists("data/parameter/time.csv"):
        for n in range(station_num):
            print('Station:', n)
            seg_slope = n_seg_slope[n]
            slope = n_slope[n]
            seg_v_lim = n_seg_v_lim[n]
            v_lim = n_v_lim[n]

            # Calculate minimum running intervals (mri) and other related values
            mri, min_t, min_seg_t, v_frontier = utils.get_mri(seg_v_lim, v_lim, seg_slope, slope, train_info)
            min_t = (min_t // 100 + 1.25) * 100  # Adjust the minimum time

            # Create a range of planned times for different scenarios
            for i in range(10):
                n_p_time.append(int(min_t + 50 * i))
                n_p_index.append(n)

        # Save planned times to CSV
        data = {'station': n_p_index, 'plan_time': n_p_time}
        df = pd.DataFrame(data)
        df.to_csv('data/parameter/time.csv', index=False)


    # Function to get grouped data for runtime plans
    def _get_grouped_data(df, group_col, data_col, stations):
        return dict(
            zip(stations, (df.groupby(group_col)[data_col].apply(list).reset_index())[data_col].tolist()))


    # Load planned times from the CSV file
    df_time = pd.read_csv('data/parameter/time.csv')
    stations = list(range(station_num))
    n_runtime = _get_grouped_data(df_time, 'station', 'plan_time', stations)

    # If MRI data doesn't exist, calculate and store for each station
    if not os.path.exists("data/mri"):
        for n in range(station_num):
            print('Station:', n)
            seg_slope = n_seg_slope[n]
            slope = n_slope[n]
            seg_v_lim = n_seg_v_lim[n]
            v_lim = n_v_lim[n]

            # Get MRI and related parameters for this station
            mri, min_t, min_seg_t, v_frontier = utils.get_mri(seg_v_lim, v_lim, seg_slope, slope, train_info)
            print('Minimum runtime:', min_t)

            # Iterate over different planned runtime scenarios
            for i in range(len(n_runtime[n])):
                print('Planned runtime', i, ':', n_runtime[n][i])

                start = time.time()

                # Plan speed intervals for energy-efficient operations
                speed_minE, speed_ub, speed_lb, minE, aetE, minE_action, minE_times, minE_step = \
                    utils.planing_speed_interval(mri, v_frontier, n_runtime[n][i],
                                                 seg_slope, slope, train_info, n_station_len[n], seg_v_lim)

                print('Actual runtime:', utils.get_line_time(speed_minE))

                # Plot the speed profiles for visualization
                plt.plot(np.arange(len(speed_lb)), speed_lb * 3.6)
                plt.plot(np.arange(len(speed_ub)), speed_ub * 3.6)
                plt.plot(np.arange(len(speed_minE)), speed_minE * 3.6)
                plt.plot(np.arange(len(mri)), mri * 3.6)
                # plt.show()  # Optionally show the plots

                # Update speed profiles to ensure they don't exceed MRI limits
                speed_minE = np.minimum(mri, speed_minE)
                speed_ub = np.minimum(mri, speed_ub)
                speed_lb = np.minimum(mri, speed_lb)

                end = time.time()
                print('Calculation took', int((end - start) / 60), 'minutes',
                      int(((end - start) / 60 - int((end - start) / 60)) * 60), 'seconds\n')

                # Store the computed values for this station and runtime
                n_mri[n].append(mri)
                n_speed_lb[n].append(speed_lb)
                n_speed_ub[n].append(speed_ub)
                n_speed_pmp[n].append(speed_minE)
                n_energy_thr[n][i] = minE
                n_energy_eff_save[n][i] = aetE
                n_energy_weight[n][i] = 3 * (minE / aetE)
                n_minE_action[n].append(minE_action)

        # Create directories and save the data
        os.makedirs('data/mri')
        os.makedirs('data/bound')
        os.makedirs('data/reward_weight')
        os.makedirs('data/expert')

        # Save dictionaries using pickle for future use
        with open('data/mri/n_mri.pickle', 'wb') as f:
            pickle.dump(n_mri, f)
        with open('data/bound/n_speed_lb.pickle', 'wb') as f:
            pickle.dump(n_speed_lb, f)
        with open('data/bound/n_speed_ub.pickle', 'wb') as f:
            pickle.dump(n_speed_ub, f)
        with open('data/bound/n_speed_pmp.pickle', 'wb') as f:
            pickle.dump(n_speed_pmp, f)
        with open('data/reward_weight/n_energy_thr.pickle', 'wb') as f:
            pickle.dump(n_energy_thr, f)
        with open('data/expert/n_minE_action.pickle', 'wb') as f:
            pickle.dump(n_minE_action, f)

    else:
        # If data already exists, load it from pickle files
        with open('data/mri/n_mri.pickle', 'rb') as f:
            n_mri = pickle.load(f)
        with open('data/bound/n_speed_lb.pickle', 'rb') as f:
            n_speed_lb = pickle.load(f)
        with open('data/bound/n_speed_ub.pickle', 'rb') as f:
            n_speed_ub = pickle.load(f)
        with open('data/bound/n_speed_pmp.pickle', 'rb') as f:
            n_speed_pmp = pickle.load(f)
