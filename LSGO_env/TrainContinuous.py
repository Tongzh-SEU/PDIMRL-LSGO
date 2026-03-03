import copy
import math
from typing import Optional
import pygame
import gym
from gym import spaces
import pandas as pd
import pickle
import numpy as np
from . import LSGO_utils as utils
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("QtAgg")   # 或 "TkAgg"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

class trainEnv(gym.Env):
    # Metadata
    metadata = {
        "render_modes": ['human', 'rgb_array'],
        "render_fps": 60
    }

    def __init__(self, task={}):
        super(trainEnv, self).__init__()

        self.is_loop = True
        self.is_lot = False
        self.loop_max = 8
        self.loop_num = 0

        self.is_over_low = False
        self.over_low_max = random.randint(1000, 5000)

        self._task = task
        self._goal = task.get('goal', np.zeros(3, dtype=np.float32))

        # Load slope, speed limits, pot, lb, ub data
        self._load_data()

        # Select, pot, lb, ub data
        """Initialize pointer variables"""
        self.loc_p, self.slope_p = self._initialize_line_params(station_seed=0, runtime_seed=1)

        self.loc_p_, self.slope_p_ = self.loc_p, self.slope_p
        # Episode length
        self.step_len = len(self.slope_x) - 1 - self.slope_p

        '''
        Reward related
        '''
        # Energy threshold
        self.e_pot_initial = utils.get_line_energy(self.pot, self.slope_x, self.slope_y, self.param)
        # Call 100 random numbers 939
        self.e_thr = self.e_pot_initial
        # Soft update e_thr
        # self.soft_update = 0.99

        # Used to interact with render
        self._initialize_info()

        # Define action and state space
        self._initialize_env_space()

        # Initialize rendering
        self._initialize_render()

        # Random seed
        self.seed()

        # Initialize environment
        self.reset()

        if self.is_render:
            self.render('human')

    def _load_data(self):
        """Load all necessary data"""
        root = 'LSGO_env/data/'
        self.df_slope = pd.read_csv(root + 'parameter/slope.csv')
        self.df_speed_lim = pd.read_csv(root + 'parameter/speed_limit.csv')
        self.df_plan_time = pd.read_csv(root + 'parameter/time.csv')
        self.param = pd.read_csv(root + 'parameter/train.csv').apply(lambda row: row.astype(float).to_dict(), axis=1).to_list()[0]

        station_num = len(self.df_slope.groupby('station'))
        self.stations = list(range(station_num))

        def _get_grouped_data(df, group_col, data_col):
            return dict(
                zip(self.stations, (df.groupby(group_col)[data_col].apply(list).reset_index())[data_col].tolist()))

        # Slope data for different lines
        self.n_slope_x = _get_grouped_data(self.df_slope, 'station', 'seg_slope')
        self.n_slope_y = _get_grouped_data(self.df_slope, 'station', 'slope')

        # Speed limit data for different lines
        self.n_v_lim_x = _get_grouped_data(self.df_speed_lim, 'station', 'seg_v_lim')
        self.n_v_lim_y = _get_grouped_data(self.df_speed_lim, 'station', 'v_lim')

        # Runtime for different lines
        self.n_runtime = _get_grouped_data(self.df_plan_time, 'station', 'plan_time')

        # Line length for different lines
        self.n_station_len = {s: seg[-1] for s, seg in self.n_v_lim_x.items()}

        with open(root + 'bound/n_speed_pmp.pickle', 'rb') as f:
            self.n_pot = pickle.load(f)
        with open(root + 'bound/n_speed_ub.pickle', 'rb') as f:
            self.n_ub = pickle.load(f)
        with open(root + 'bound/n_speed_lb.pickle', 'rb') as f:
            self.n_lb = pickle.load(f)
        with open(root + 'mri/n_mri.pickle', 'rb') as f:
            self.n_mrt = pickle.load(f)

    def _initialize_line_params(self, station_seed, runtime_seed):
        """Initialize train operation parameters"""
        # Planned time
        self.runtime = self.n_runtime[station_seed][runtime_seed]
        # Line length
        self.line_len = self.n_station_len[station_seed]

        # Slope parameters (left-open, right-closed)
        self.slope_y = self.n_slope_y[station_seed]
        self.slope_x = self.n_slope_x[station_seed]

        # Speed limit parameters (left-open, right-closed)
        self.v_lim_y = self.n_v_lim_y[station_seed]
        self.v_lim_x = self.n_v_lim_x[station_seed]

        # Shortest runtime, trajectory to optimize, and upper/lower boundary constraints
        self.mrt = self.n_mrt[station_seed][runtime_seed]
        self.pot = self.n_pot[station_seed][runtime_seed]
        self.lb = self.n_lb[station_seed][runtime_seed]
        self.ub = self.n_ub[station_seed][runtime_seed]

        self.speed = self.pot
        self.lot = self.pot

        self.pot_initialize = self.pot
        self.pot_acc_initialize = utils.get_line_action(self.pot, self.slope_x, self.slope_y, self.param)
        self.pot_time_initialize = utils.get_line_step_time(self.pot)

        # Calculate the frequency of each value
        unique, counts = np.unique(self.pot, return_counts=True)
        # Find the most frequent value
        most_frequent_value = unique[np.argmax(counts)]

        # Find the last occurrence of the value
        self.coasting_loc = int((np.max(np.where(self.pot == most_frequent_value)) + self.line_len) / 2)

        self.start_loc = np.argmax(self.lb)
        slope_p = 0
        for s_i in range(len(self.slope_x)):
            if self.start_loc < self.slope_x[s_i + 1]:
                break
            else:
                slope_p += 1
        self.start_loc = self.slope_x[slope_p]
        return self.start_loc, slope_p

    def _initialize_env_space(self):
        # Normalization standards
        self.t_nor = 600
        self.e_nor = 1000
        self.v_nor = 310 / 3.6
        self.s_y_nor = 10
        self.s_x_nor = 1e5

        # Observation range
        self.ob_scope = 4
        self.state_num = 8 + self.ob_scope * 2
        self.info_num = 7

        self.state = np.zeros((self.state_num, self.step_len + 1))
        self.info = np.zeros((self.info_num, self.step_len + 1))

        """Define state space"""
        # 0: Original time consumption for this slope segment /600, 1: Time error with planned time t/600,
        # 2: Original energy consumption for this slope segment e/1000,
        # 3: Remaining estimated energy consumption e/1000, 4: Current position x/len, 5: Current speed v/310/3.6
        # 6: Current switch point s_p, 7: Front power f_p, 8: Rear power b_p
        # 9: Basic traction to maintain current speed, 10: Slope length x/(1e3+2), 11: Slope g/10
        min_state_values = [0, -6, 0, -30, 0, 0, 0, -2]
        max_state_values = [6, 6, 10, 30, 1, 1, 1, 2]

        for _ in range(self.ob_scope):
            min_state_values.extend([0, -1])
            max_state_values.extend([5, 1])

        self.low_state = np.array(min_state_values)
        self.high_state = np.array(max_state_values)

        # Observation space range
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # Switch point, front power, rear power
        min_action_values = [0, 0, 0]
        max_action_values = [1, 1, 1]

        self.low_action = np.array(min_action_values)
        self.high_action = np.array(max_action_values)

        # Action space range
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, shape=(3,), dtype=np.float32)

    def _initialize_info(self):
        self.time_pot = np.array(utils.get_line_step_time(self.pot))
        self.time = copy.deepcopy(self.time_pot)
        self.energy_pot = np.array(utils.get_line_step_energy(self.pot, self.slope_x, self.slope_y, self.param))
        self.energy = copy.deepcopy(self.energy_pot)
        self.acc_pot = utils.get_line_action(self.pot, self.slope_x, self.slope_y, self.param)
        self.acc = copy.deepcopy(self.acc_pot)

    def _initialize_render(self):
        self.render_style = 'plt'
        self.is_render = True
        # Display calculation points separated by render
        self.render_gap = self.line_len
        # Fill to render precision
        self.gap_num = 100

        self.lb_render_x, self.lb_render_y = utils.get_render(self.lb, self.gap_num)
        self.ub_render_x, self.ub_render_y = utils.get_render(self.ub, self.gap_num)
        self.mrt_render_x, self.mrt_render_y = utils.get_render(self.mrt, self.gap_num)
        self.v_lim_render_x, self.v_lim_render_y = utils.get_v_lim_render(self.v_lim_x, self.v_lim_y)
        self.slope_render_x, self.slope_render_y = utils.get_v_lim_render(self.slope_x, self.slope_y)
        if self.is_render:
            if self.render_style == 'pygame':
                pygame.init()
                pygame.font.init()
                self.screen = None
                self.clock = None
                self.is_open = True
                self.surf = None
                self.speed_h = 3 / 4
                self.speed_w = 2 / 3
                self.energy_h = 1 / 2
                self.energy_w = 1 / 3
                self.t_error_h = 1 / 2
                self.t_error_w = 1 / 3
                self.slope_h = 1 / 4
                self.slope_w = 2 / 3
                self.screen_W = int(1920 * 0.75)
                self.screen_H = int(1080 * 0.75)
            else:
                plt.ion()  # Enable interactive mode

                self.gym_fig = plt.figure(figsize=(12, 6.75))
                self.gs = GridSpec(5, 2, figure=self.gym_fig, width_ratios=[10, 5], height_ratios=[12, 0.25, 5, 0.25, 5],
                                   left=0.05, right=0.98, top=0.95, bottom=0.05)

                # Create subplots and assign them to different grids
                self.ax_speed = self.gym_fig.add_subplot(self.gs[0, 0])
                self.ax_acc = self.gym_fig.add_subplot(self.gs[2, 0])
                self.ax_slope = self.gym_fig.add_subplot(self.gs[4, 0])
                self.ax_energy = self.gym_fig.add_subplot(self.gs[0, 1])
                self.ax_t_error = self.gym_fig.add_subplot(self.gs[2:5, 1])

                # Set titles for subplots
                self.ax_speed.set_title('Speed Profiles')
                self.ax_acc.set_title('Train Gear Action')
                self.ax_slope.set_title('Slope Profile')
                self.ax_energy.set_title('Energy Profiles')
                self.ax_t_error.set_title('Run-time Error with POT')

                self.ax_speed.grid()
                self.ax_acc.grid()
                self.ax_slope.grid()
                self.ax_energy.grid()
                self.ax_t_error.grid()

                self.ax_speed.set_xlim(-500, self.line_len + 500)
                self.ax_acc.set_xlim(-500, self.line_len + 500)
                self.ax_slope.set_xlim(-500, self.line_len + 500)
                self.ax_energy.set_xlim(-500, self.line_len + 500)
                self.ax_t_error.set_xlim(-500, self.line_len + 500)

                self.ax_speed.set_ylim(0, 320)
                self.ax_acc.set_ylim(-1.1, 1.1)
                self.ax_slope.set_ylim(-15, 15)

                self.ax_speed.set_ylabel('Speed [km/h]')
                self.ax_acc.set_ylabel('Gear action [%]')
                self.ax_slope.set_ylabel('Slope [‰]')
                self.ax_energy.set_ylabel('Energy [kWh]')
                self.ax_t_error.set_ylabel('T_error [s]')

                # ax_speed
                # Render speed curve in the subwindow
                self.l_ub, = self.ax_speed.plot([], color='red', label='ub')
                self.l_lb, = self.ax_speed.plot([], color='blue', label='lb')
                self.l_mrt, = self.ax_speed.plot([], color='gray', label='mrt')
                self.l_v_lim, = self.ax_speed.plot([], '--', color='black', label='v_lim')
                self.l_pot, = self.ax_speed.plot([], color='orange', label='pot')
                self.l_speed, = self.ax_speed.plot([], color='green', label='speed')
                self.l_pot_loc = self.ax_speed.scatter(x=0, y=0, color='orange', s=20, marker='o')
                self.l_speed_loc = self.ax_speed.scatter(x=0, y=0, color='green', s=20, marker='p')

                # ax_gear_action
                self.l_pot_acc, = self.ax_acc.plot([], color='orange', label='pot_acc')
                self.l_speed_acc, = self.ax_acc.plot([], color='green', label='speed_acc')
                self.l_pot_acc_loc = self.ax_acc.scatter(x=0, y=0, color='orange', s=20, marker='o')
                self.l_speed_acc_loc = self.ax_acc.scatter(x=0, y=0, color='green', s=20, marker='p')

                # ax_energy
                self.l_pot_e, = self.ax_energy.plot([], color='orange', label='pot_e')
                self.l_speed_e, = self.ax_energy.plot([], color='green', label='speed_e')
                self.l_pot_e_loc = self.ax_energy.scatter(x=0, y=0, color='orange', s=20, marker='o')
                self.l_speed_e_loc = self.ax_energy.scatter(x=0, y=0, color='green', s=20, marker='p')

                # ax_slope
                self.l_slope, = self.ax_slope.plot([], color='black', label='slope')
                self.l_slope_shift = self.ax_slope.scatter(x=0, y=0, color='red', s=20, marker='*')
                self.l_slope_loc = self.ax_slope.scatter(x=0, y=0, color='black', s=20, marker='o')

                # ax_t_error
                self.l_t_error, = self.ax_t_error.plot([], color='black', label='t_error')
                self.l_t_error_loc = self.ax_t_error.scatter(x=0, y=0, color='black', s=20, marker='o')

                self.ax_speed.legend()
                self.ax_acc.legend()
                self.ax_slope.legend()
                self.ax_energy.legend()
                self.ax_t_error.legend()

    # Random seed
    def seed(self, seed=None):
        pass

    def get_max_step_len(self):
        return self.step_len

    def sample_tasks(self, num_tasks):
        max_num_tasks = len(self.df_plan_time['plan_time'])
        assert num_tasks <= max_num_tasks, 'num_tasks > max_num_tasks'
        goals_index = []
        for key, value_list in self.n_runtime.items():
            for index in range(len(value_list)):
                goals_index.append([key, index])
        goals = np.hstack((np.array(goals_index), np.array(self.df_plan_time['plan_time']).reshape(-1, 1)))
        chosen_indices = np.random.choice(goals.shape[0], num_tasks, replace=False)
        goals = goals[chosen_indices]
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']
        """Initialize pointer variables"""
        self.loc_p, self.slope_p = self._initialize_line_params(station_seed=self._goal[0], runtime_seed=self._goal[1])
        self.loc_p_, self.slope_p_ = self.loc_p, self.slope_p
        # Episode length
        self.step_len = len(self.slope_x) - 1 - self.slope_p

        '''
        Reward related
        '''
        # Energy threshold
        self.e_pot_initial = utils.get_line_energy(self.pot, self.slope_x, self.slope_y, self.param)
        # Call 100 random numbers 939
        self.e_thr = self.e_pot_initial
        # Soft update e_thr
        # self.soft_update = 0.99

        # Used to interact with render
        self._initialize_info()

        # Define action and state space
        self._initialize_env_space()

        # Initialize rendering
        self._initialize_render()

        # Random seed
        self.seed()

        # Initialize environment
        state = self.reset()

        return state


    # TODO: Needs adjustment
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None):

        self.loc_p, self.slope_p = self.loc_p_, self.slope_p_

        # If not looped, reset at the end of each episode
        if not self.is_loop:
            self.pot = copy.deepcopy(self.pot_initialize)
            self.speed = copy.deepcopy(self.pot_initialize)
        else:
            if self.loop_num != self.loop_max:
                self.pot = copy.deepcopy(self.speed)
                self.speed = copy.deepcopy(self.speed)
            else:
                self.loop_num = 0
                if self.is_lot:
                    self.pot = copy.deepcopy(self.lot)
                    self.speed = copy.deepcopy(self.lot)
                else:
                    self.pot = copy.deepcopy(self.pot_initialize)
                    self.speed = copy.deepcopy(self.pot_initialize)

        self.acc_pot = utils.get_line_action(self.pot, self.slope_x, self.slope_y, self.param)
        self.acc = copy.deepcopy(self.acc_pot)
        self.time_pot = utils.get_line_step_time(self.pot)
        self.time = copy.deepcopy(self.time_pot)
        self.energy_pot = utils.get_line_step_energy(self.pot, self.slope_x, self.slope_y, self.param)
        self.energy = copy.deepcopy(self.energy_pot)
        if self.is_render:
            self.pot_render_x, self.pot_render_y = utils.get_render(self.pot, self.gap_num)
            self.speed_render_x, self.speed_render_y = utils.get_render(self.speed, self.gap_num)

        self.state.fill(0)
        self.info.fill(0)

        """Define state space"""
        # 0: Original time consumption for this slope segment t/600, 1: Time error with planned time t/600,
        # 2: Original energy consumption for this slope segment e/1000,
        # 3: Remaining estimated energy consumption e/1000, 4: Current position x/len, 5: Current speed v/310/3.6
        # 6: Next switch speed v/310/3.6, 7: Next lb v/310/3.6, 8: Next ub v/310/3.6
        # 9: Basic traction to maintain current speed, 10: Slope length x/(1e3+2), 11: Slope g/10
        self.state[0][0] = utils.get_seg_time(self.speed, self.loc_p, self.slope_x[self.slope_p + 1]) / self.t_nor
        self.state[1][0] = 0
        self.state[2][0] = utils.get_seg_energy(self.speed, self.loc_p, self.slope_x[self.slope_p + 1], self.slope_x,
                                                self.slope_y, self.param) / self.e_nor
        self.state[3][0] = self.e_pot_initial / self.e_nor
        self.state[4][0] = self.loc_p / self.line_len
        self.state[5][0] = self.pot[self.loc_p] / self.v_nor
        self.state[6][0] = self.pot[self.slope_x[self.slope_p + 1]] / self.v_nor

        # Calculate the slope acceleration at the current position
        slope_acc = utils.get_slope_accelerated(self.loc_p, self.slope_x, self.slope_y, self.param)
        # Calculate the basic resistance acceleration
        basic_acc = (self.param['drag_coefficient_a'] + self.param['drag_coefficient_b'] * self.pot[
            self.loc_p - 1] * 3.6
                     + self.param['drag_coefficient_c'] * (self.pot[self.loc_p - 1] * 3.6) ** 2) / self.param['mass']
        self.state[7][0] = -(slope_acc + basic_acc)
        for ob_i in range(self.ob_scope):
            if self.slope_p + ob_i < len(self.slope_x) - 1:
                self.state[8 + ob_i * 2][0] = (self.slope_x[self.slope_p + ob_i + 1] - self.slope_x[
                    self.slope_p + ob_i]) / self.s_x_nor
                self.state[9 + ob_i * 2][0] = (self.slope_y[self.slope_p + ob_i]) / self.s_y_nor
            else:
                self.state[8 + ob_i * 2][0] = 0
                self.state[9 + ob_i * 2][0] = 0

        return np.array([index[0] for index in self.state], dtype=np.float32)



    def close(self):
        if self.is_render:
            plt.close(self.gym_fig)

    def step(self, agent_action):
        shift_p = int(agent_action[0] * (self.slope_x[self.slope_p + 1] - self.slope_x[self.slope_p])) + self.loc_p
        front_pct = agent_action[1]
        back_pct = agent_action[2]

        # if self.is_render:
        #    self.render()

        # self.speed_action[0][self.slope_p - self.slope_p_] = agent_action[0]
        # self.speed_action[1][self.slope_p - self.slope_p_] = agent_action[1]
        # self.speed_action[2][self.slope_p - self.slope_p_] = agent_action[2]

        # The loop will continue until the current position exceeds the position of the next slope
        while self.loc_p < self.slope_x[self.slope_p + 1]:
            # Increase the current position by 1
            self.loc_p += 1

            # Get the speed of the previous position
            last_v = self.speed[self.loc_p - 1]

            # Calculate the maximum traction, which is the smaller of the traction and power multiplied by the vehicle mass and the slope
            max_train_F = min(self.param['traction_140'], self.param['power'] / (
                    (1 + self.param['mass_factor']) * self.param['mass'] * last_v))
            # If the current position is less than or equal to the transition point, the calculated acceleration will be based on the forward percentage
            if self.loc_p <= shift_p:
                train_acc = front_pct * max_train_F
            else:
                # Otherwise, acceleration will be based on the rear percentage
                train_acc = back_pct * max_train_F

            if self.lb[self.loc_p] == self.ub[self.loc_p] and self.ub[self.loc_p] - self.ub[self.loc_p - 1] > 0:
                train_acc = max_train_F
            if self.loc_p >= self.coasting_loc and self.lb[self.loc_p] != self.ub[self.loc_p]:
                train_acc = 0

            # Calculate the slope acceleration of the current position
            slope_acc = utils.get_slope_accelerated(self.loc_p, self.slope_x, self.slope_y, self.param)
            # Calculate basic drag acceleration
            basic_acc = (self.param['drag_coefficient_a'] + self.param['drag_coefficient_b'] * last_v * 3.6
                         + self.param['drag_coefficient_c'] * (last_v * 3.6) ** 2) / self.param['mass']
            # Use the dynamics formula to calculate the current speed
            now_v = utils.get_speed(last_v, train_acc - slope_acc - basic_acc)


            # If the calculated speed exceeds the maximum speed limit, set it to the maximum/minimum speed limit
            if now_v > self.mrt[self.loc_p]:
                now_v = self.mrt[self.loc_p]
                # Recalculate acceleration to ensure that the speed does not exceed the maximum/minimum speed
                train_acc = (now_v ** 2 - last_v ** 2) / 2 + slope_acc + basic_acc
            if self.lb[self.loc_p] == self.ub[self.loc_p] and now_v >= self.ub[self.loc_p]:
                now_v = self.mrt[self.loc_p]
                train_acc = (now_v ** 2 - last_v ** 2) / 2 + slope_acc + basic_acc
            else:
                if self.is_over_low:
                    train_acc = max_train_F * random.uniform(0, 1)
                    now_v = utils.get_speed(last_v, train_acc - slope_acc - basic_acc)
                    self.over_low_max -= 1
                    if self.over_low_max == 0:
                        self.is_over_low = False
                else:
                    if now_v < 60 / 3.6:
                        # now_v = 80/3.6
                        # train_acc = (now_v ** 2 - last_v ** 2) / 2 + slope_acc + basic_acc
                        train_acc = max_train_F
                        now_v = utils.get_speed(last_v, train_acc - slope_acc - basic_acc)
                        self.is_over_low = True
                        self.over_low_max = random.randint(100, 500)

            # Get the current execution time
            step_time = utils.get_move_time(now_v, last_v, train_acc - slope_acc - basic_acc)
            # energy
            step_energy = utils.get_energy(train_acc, self.param)

            # Store the speed, time, and energy of each step
            self.speed[self.loc_p] = now_v
            self.time[self.loc_p] = self.time[self.loc_p - 1] + step_time
            self.energy[self.loc_p] = self.energy[self.loc_p - 1] + step_energy
            self.acc[self.loc_p] = train_acc / max_train_F

            if self.is_render:
                if self.loc_p % self.render_gap == 0:
                    self.render('human')

        # Update optimized line
        loc_temp = self.loc_p
        while self.speed[loc_temp] != self.pot[loc_temp]:
            loc_temp += 1
            last_v_temp = self.speed[loc_temp - 1]

            # Calculate the slope acceleration at the current position
            slope_acc_temp = utils.get_slope_accelerated(loc_temp, self.slope_x, self.slope_y, self.param)
            # Calculating Basic Drag Acceleration
            basic_acc_temp = (self.param['drag_coefficient_a'] + self.param['drag_coefficient_b'] * last_v_temp * 3.6
                              + self.param['drag_coefficient_c'] * (last_v_temp * 3.6) ** 2) / self.param['mass']
            # Calculates the maximum tractive effort, which is the smaller of tractive effort and power multiplied by vehicle mass and slope
            max_train_F_temp = min(self.param['traction_140'], self.param['power'] / (
                    (1 + self.param['mass_factor']) * self.param['mass'] * last_v_temp))
            # End point speed, if less than the original pot, accelerate, greater than the original pot, glide
            if self.speed[loc_temp - 1] < self.pot[loc_temp - 1]:
                train_acc_temp = 1 * max_train_F_temp
                now_v_temp = utils.get_speed(last_v_temp, train_acc_temp - slope_acc_temp - basic_acc_temp)
                if now_v_temp > self.pot[loc_temp]:
                    now_v_temp = self.pot[loc_temp]
            else:
                # Otherwise, the acceleration will be based on the rear percentage
                train_acc_temp = 0 * max_train_F_temp
                now_v_temp = utils.get_speed(last_v_temp, train_acc_temp - slope_acc_temp - basic_acc_temp)

                # If the calculated speed exceeds the maximum speed limit, set the maximum/minimum speed limit
                if now_v_temp > self.mrt[loc_temp]:
                    now_v_temp = self.mrt[loc_temp]
                    # Recalculate acceleration to ensure that speed does not exceed maximum/minimum speed
                    train_acc_temp = (now_v_temp ** 2 - last_v_temp ** 2) / 2 + slope_acc_temp + basic_acc_temp
                if self.lb[loc_temp] == self.ub[loc_temp]:
                    now_v_temp = self.mrt[loc_temp]
                    train_acc_temp = (now_v_temp ** 2 - last_v_temp ** 2) / 2 + slope_acc_temp + basic_acc_temp
                else:
                    if now_v_temp < 60 / 3.6:
                        # now_v_temp = 80 / 3.6
                        train_acc_temp = max_train_F_temp
                        now_v_temp = utils.get_speed(last_v_temp, train_acc_temp - slope_acc_temp - basic_acc_temp)

                '''
                if now_v_temp > self.ub[loc_temp]:
                    now_v_temp = self.ub[loc_temp]
                    train_acc_temp = (now_v_temp ** 2 - last_v_temp ** 2) / 2 + slope_acc_temp + basic_acc_temp
                if now_v_temp < self.pot[loc_temp]:
                    now_v_temp = self.pot[loc_temp]
                '''
            step_time = utils.get_move_time(now_v_temp, last_v_temp, train_acc_temp - slope_acc_temp - basic_acc_temp)
            step_energy = utils.get_energy(train_acc_temp, self.param)

            self.speed[loc_temp] = now_v_temp
            self.time[loc_temp] = self.time[loc_temp - 1] + step_time
            self.energy[loc_temp] = self.energy[loc_temp - 1] + step_energy

            if self.is_render:
                if loc_temp % self.render_gap == 0:
                    self.render('human')
        # self.time[loc_temp:] = self.time_pot[loc_temp:]
        # self.energy[loc_temp:] = self.energy_pot[loc_temp:]
        # self.time[loc_temp:] = self.time[loc_temp:] + self.time[loc_temp - 1] - self.time_pot[loc_temp - 1]
        # self.energy[loc_temp:] = self.energy[loc_temp:] + self.energy[loc_temp - 1] - self.energy_pot[loc_temp - 1]

        # Train jerk
        now_jerk = utils.get_line_jerk(self.speed)

        # Reward Settlement
        done = bool(self.loc_p == self.line_len)

        terminal = False
        if not self.is_loop and done:
            terminal = True

        if self.is_loop and done:
            self.loop_num += 1
            if self.loop_num == self.loop_max:
                terminal = True

        if done:
            if self.is_loop:
                self.pot = self.speed
                self.energy_pot = self.energy
                self.time_pot = self.time

        # Update status of next step
        self.slope_p += 1
        # Update status
        # 0 Original time consumed for this slope segment t/600 1 Error with planned time t/600 2 Original energy consumed for this slope segment e/1000
        # 3 Remaining estimated energy consumption e/1000 4 Current position x/len 5 Current speed v/310/3.6
        # 6 Next transition point speed v/310/3.6 #7 Next lb v/310/3.6 #8 Next ub v/310/3.6
        # 9 Basic traction to maintain current speed 10 Slope length x/(1e3+2) 11 Slope g/10
        if not done:
            self.state[0][self.slope_p - self.slope_p_] = utils.get_seg_time(self.speed, self.slope_x[self.slope_p],
                                                                             self.slope_x[
                                                                                 self.slope_p + 1]) / self.t_nor
            self.state[1][self.slope_p - self.slope_p_] = (self.runtime - utils.get_line_time(self.speed)) / self.t_nor
            self.state[2][self.slope_p - self.slope_p_] = utils.get_seg_energy(self.speed, self.slope_x[self.slope_p],
                                                                               self.slope_x[self.slope_p + 1],
                                                                               self.slope_x, self.slope_y,
                                                                               self.param) / self.e_nor
            self.state[3][self.slope_p - self.slope_p_] = (self.e_pot_initial - self.energy[self.loc_p]) / self.e_nor
            self.state[4][self.slope_p - self.slope_p_] = self.loc_p / self.line_len
            self.state[5][self.slope_p - self.slope_p_] = self.speed[self.loc_p] / self.v_nor
            self.state[6][self.slope_p - self.slope_p_] = self.speed[self.slope_x[self.slope_p + 1]] / self.v_nor
            # self.state[7][self.slope_p - self.slope_p_] = self.lb[self.slope_x[self.slope_p + 1]] / self.v_nor
            # self.state[8][self.slope_p - self.slope_p_] = self.ub[self.slope_x[self.slope_p + 1]] / self.v_nor
            slope_acc_avg = self.slope_y[self.slope_p] * 9.81 / 1000
            avg_v = np.mean(self.speed[self.slope_x[self.slope_p]:self.slope_x[self.slope_p + 1]])
            basic_acc_avg = (self.param['drag_coefficient_a'] + self.param['drag_coefficient_b'] * avg_v * 3.6
                             + self.param['drag_coefficient_c'] * (avg_v * 3.6) ** 2) / self.param['mass']
            max_train_F_avg = min(self.param['traction_140'], self.param['power'] / (
                    (1 + self.param['mass_factor']) * self.param['mass'] * avg_v))
            self.state[7][self.slope_p - self.slope_p_] = (slope_acc_avg + basic_acc_avg) / max_train_F_avg
            for ob_i in range(self.ob_scope):
                if self.slope_p + ob_i < len(self.slope_x) - 1:
                    self.state[8 + ob_i * 2][self.slope_p - self.slope_p_] = (self.slope_x[self.slope_p + ob_i + 1] -
                                                                              self.slope_x[
                                                                                  self.slope_p + +ob_i]) / self.s_x_nor
                    self.state[9 + ob_i * 2][self.slope_p - self.slope_p_] = (self.slope_y[
                        self.slope_p + +ob_i]) / self.s_y_nor
                else:
                    self.state[8 + ob_i * 2][self.slope_p - self.slope_p_] = 0
                    self.state[9 + ob_i * 2][self.slope_p - self.slope_p_] = 0
            self.info[0, self.slope_p - self.slope_p_] = utils.get_line_energy(self.speed, self.slope_x, self.slope_y,
                                                                               self.param)
            self.info[1, self.slope_p - self.slope_p_] = utils.get_line_time(self.speed) - self.runtime
            self.info[2, self.slope_p - self.slope_p_] = now_jerk
            speed_copy = copy.deepcopy(self.speed)
        else:
            self.state[0:5, -1] = [0, (self.time_pot[self.loc_p] - self.time[self.loc_p]) / self.t_nor,
                                   0, (self.e_pot_initial - self.energy[self.loc_p]) / self.e_nor, 1]
            self.info[0, -1] = utils.get_line_energy(self.speed, self.slope_x, self.slope_y, self.param)
            self.info[1, -1] = utils.get_line_time(self.speed) - self.runtime
            self.info[2, -1] = now_jerk
            speed_copy = copy.deepcopy(self.speed)

        j_w = -1 / 8
        discount_w = utils.discount_w(self.loc_p, self.line_len) * 0.1
        if done:
            discount_w = 1

        jerk_reward = j_w * now_jerk

        # On-time rewards
        time_reward = (3 - (3 / 12) * abs(self.info[1, self.slope_p - self.slope_p_]))
        if time_reward >= 0:
            time_reward *= 10

        # if self.info[0, self.slope_p - self.slope_p_] >= self.e_pot_initial:
        # energy_reward = min(discount_w * (3 / 10.68) * (self.e_pot_initial - self.energy[-1]), 6) * 10
        '''
        if self.info[0, self.slope_p - self.slope_p_] >= self.e_pot_initial:
            energy_reward = min(discount_w * (-3 + (6 / 10.68) * (self.e_pot_initial - self.info[0, self.slope_p - self.slope_p_])), 6)
            if time_reward >= 0:
                time_reward = 0
        else:
            energy_reward = min(discount_w * (3 / 10.68) * (self.e_pot_initial - self.info[0, self.slope_p - self.slope_p_]), 6) * 10
            if time_reward <= 0:
               energy_reward = 0 
        '''
        if self.info[0, self.slope_p - self.slope_p_] >= self.e_pot_initial:
            energy_reward = min((-3 + (6 / 10.68) * (self.e_pot_initial - self.info[0, self.slope_p - self.slope_p_])),
                                6)
            if time_reward >= 0:
                time_reward = 0
        else:
            energy_reward = min((3 / 10.68) * (self.e_pot_initial - self.info[0, self.slope_p - self.slope_p_]), 6) * 10
            if time_reward <= 0:
                energy_reward = 0

        # print(f'{self.slope_p-self.slope_p_}: energy_save:{ (self.e_pot_initial - self.energy[-1])} time_error:{abs(self.runtime - self.time[-1])}')

        '''
        if time_reward > 0 and energy_reward > 0:
            energy_reward = energy_reward*(math.tanh(20/abs(time_reward**2-energy_reward**2))+1)
            time_reward = time_reward*(math.tanh(20/abs(time_reward**2-energy_reward**2))+1)
        '''

        if self.is_lot:
            if energy_reward > 0 and time_reward >= 2.92:
                if utils.get_line_energy(self.speed, self.slope_x, self.slope_y, self.param) < utils.get_line_energy(
                        self.lot, self.slope_x, self.slope_y, self.param):
                    self.lot = self.speed

        '''
        if sum_energy < self.e_thr and self.e_thr >= self.e_pot_initial and abs((self.state[1][self.slope_p-self.slope_p_]*self.t_nor)) < 12:
            e_loss = utils.t_error_e_loss(self.pot, self.state[1][self.slope_p-self.slope_p_]*self.t_nor, self.param, self.slope_x, self.slope_y)
            if sum_energy+e_loss < self.e_pot_initial:
                self.e_thr = self.e_thr * self.soft_update + (self.energy[self.loc_p]+e_loss) * (1 - self.soft_update)
        '''

        if done:
            self.info[3, -1] = energy_reward
            self.info[4, -1] = time_reward
            self.info[5, -1] = jerk_reward
            self.info[6, -1] = self.e_thr
        else:
            self.info[3, self.slope_p - self.slope_p_] = energy_reward
            self.info[4, self.slope_p - self.slope_p_] = time_reward
            self.info[5, self.slope_p - self.slope_p_] = jerk_reward
            self.info[6, self.slope_p - self.slope_p_] = self.e_thr
        reward = jerk_reward
        if energy_reward < 0 and time_reward < 0:
            reward = jerk_reward + -math.sqrt(time_reward * energy_reward)
        if energy_reward == 0 and time_reward < 0:
            reward = jerk_reward + time_reward
        if time_reward == 0 and energy_reward < 0:
            reward = jerk_reward + energy_reward
        if energy_reward > 0 and time_reward > 0:
            reward = jerk_reward + math.sqrt(time_reward * energy_reward)
        reward = reward * discount_w

        state = np.array([index[self.slope_p - self.slope_p_] for index in self.state], dtype=np.float32)
        info = np.array([index[self.slope_p - self.slope_p_] for index in self.info], dtype=np.float32)
        infos = {'info': info,
                 'speed': speed_copy
                 }
        return state, reward, done, terminal, infos

    def render(self, mode='human'):
        if self.render_style == 'pygame':
            # Calculate the dimensions of sub-windows
            dimensions = {
                "speed": (self.screen_W * self.speed_w, self.screen_H * self.speed_h),
                "energy": (self.screen_W * self.energy_w, self.screen_H * self.energy_h),
                "t_error": (self.screen_W * self.t_error_w, self.screen_H * self.t_error_h),
                "slope": (self.screen_W * self.slope_w, self.screen_H * self.slope_h)
            }

            # Define colors
            colors = {
                "gray": (127, 127, 127),
                "orange": (228, 139, 33),
                "red": (220, 73, 11),
                "blue": (81, 105, 222),
                "green": (84, 139, 49),
                "black": (0, 0, 0),
                "bg_speed": (238, 197, 133),
                "bg_energy": (243, 210, 201),
                "bg_slope": (219, 229, 254),
                "bg_t_error": (221, 244, 213)
            }

            # Initialize font
            my_font = pygame.font.SysFont("pingfang", 16)

            # Initialize screen and clock
            if self.screen is None:
                self.screen = pygame.display.set_mode((self.screen_W, self.screen_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Create sub-windows
            sub_windows = {
                "speed": pygame.Surface(dimensions["speed"]),
                "energy": pygame.Surface(dimensions["energy"]),
                "slope": pygame.Surface(dimensions["slope"]),
                "t_error": pygame.Surface(dimensions["t_error"])
            }

            # Fill background color in sub-windows
            sub_windows["speed"].fill(colors["bg_speed"])
            sub_windows["energy"].fill(colors["bg_energy"])
            sub_windows["slope"].fill(colors["bg_slope"])
            sub_windows["t_error"].fill(colors["bg_t_error"])

            # Render titles in sub-windows
            sub_windows["speed"].blit(my_font.render("Optimized Trajectory Curve", True, colors["black"]), (5, 5))
            sub_windows["energy"].blit(my_font.render("Energy Comparison Curve", True, colors["black"]), (5, 5))
            sub_windows["slope"].blit(my_font.render("Slope Position Curve", True, colors["black"]), (5, 5))
            sub_windows["t_error"].blit(my_font.render("On-time Error Curve", True, colors["black"]), (5, 5))

            # Function to render lines in sub-windows
            def render_lines(sub_window, lines_data):
                for line_data, color in lines_data:
                    pygame.draw.aalines(sub_window, color=color, points=line_data, closed=False)

            # Render speed curve in the sub-window
            speed_lines = [
                (utils.get_line_to_pygame(
                    np.concatenate([np.arange(0, len(self.speed), self.gap_num), np.array([self.line_len])]),
                    self.speed[::self.gap_num] + [self.speed[-1]],
                    self.line_len, 310 / 3.6,
                    dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                    dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05), colors["green"]),
                (utils.get_line_to_pygame(self.lb_render_x, self.lb_render_y, self.line_len, 310 / 3.6,
                                          dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                                          dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05),
                 colors["blue"]),
                (utils.get_line_to_pygame(self.ub_render_x, self.ub_render_y, self.line_len, 310 / 3.6,
                                          dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                                          dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05), colors["red"]),
                (utils.get_line_to_pygame(
                    np.concatenate([np.arange(0, len(self.speed), self.gap_num), np.array([self.line_len])]),
                    self.pot[::self.gap_num] + [self.pot[-1]],
                    self.line_len, 310 / 3.6,
                    dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                    dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05), colors["orange"]),
                (utils.get_line_to_pygame(self.v_lim_render_x, self.v_lim_render_y, self.line_len, 310,
                                          dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                                          dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05),
                 colors["black"]),
                (utils.get_line_to_pygame(self.mrt_render_x, self.mrt_render_y, self.line_len, 310 / 3.6,
                                          dimensions["speed"][0] * 0.96, dimensions["speed"][1] * 0.9,
                                          dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05),
                 colors["gray"]),
            ]
            render_lines(sub_windows["speed"], speed_lines)

            # Render x-y axis
            pygame.draw.line(sub_windows["speed"], colors["black"],
                             (dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.05),
                             (dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.95), 1)
            pygame.draw.line(sub_windows["speed"], colors["black"],
                             (dimensions["speed"][0] * 0.02, dimensions["speed"][1] * 0.95),
                             (dimensions["speed"][0] * 0.98, dimensions["speed"][1] * 0.95), 1)
            pygame.draw.circle(sub_windows["speed"], colors["red"],
                               (2 + dimensions["speed"][0] * 0.02 + dimensions["speed"][
                                   0] * 0.96 * self.loc_p / self.line_len,
                                2 + dimensions["speed"][1] * 0.05 + dimensions["speed"][1] * 0.9 * (
                                            1 - (self.speed[self.loc_p] / (310 / 3.6)))), 3)
            pygame.draw.circle(sub_windows["speed"], colors["red"],
                               (2 + dimensions["speed"][0] * 0.02 + dimensions["speed"][
                                   0] * 0.96 * self.loc_p / self.line_len,
                                2 + dimensions["speed"][1] * 0.05 + dimensions["speed"][1] * 0.9 * (
                                            1 - (self.pot[self.loc_p] / (310 / 3.6)))), 3)

            # Render energy curve in the sub-window
            energy_lines = [
                (utils.get_line_to_pygame(
                    np.concatenate([np.arange(0, len(self.energy), self.gap_num), np.array([self.line_len])]),
                    self.energy[::self.gap_num] + [self.energy[-1]], self.line_len, self.e_pot_initial,
                    dimensions["energy"][0] * 0.96, dimensions["energy"][1] * 0.9,
                    dimensions["energy"][0] * 0.02, dimensions["energy"][1] * 0.05), colors["green"]),
                (utils.get_line_to_pygame(
                    np.concatenate([np.arange(0, len(self.energy_pot), self.gap_num), np.array([self.line_len])]),
                    self.energy_pot[::self.gap_num] + [self.energy_pot[-1]], self.line_len, self.e_pot_initial,
                    dimensions["energy"][0] * 0.96, dimensions["energy"][1] * 0.9,
                    dimensions["energy"][0] * 0.02, dimensions["energy"][1] * 0.05), colors["orange"]),
            ]
            render_lines(sub_windows["energy"], energy_lines)

            # Render x-y axis for energy window
            pygame.draw.line(sub_windows["energy"], colors["black"],
                             (dimensions["energy"][0] * 0.02, dimensions["energy"][1] * 0.05),
                             (dimensions["energy"][0] * 0.02, dimensions["energy"][1] * 0.95), 1)
            pygame.draw.line(sub_windows["energy"], colors["black"],
                             (dimensions["energy"][0] * 0.02, dimensions["energy"][1] * 0.95),
                             (dimensions["energy"][0] * 0.98, dimensions["energy"][1] * 0.95), 1)
            pygame.draw.circle(sub_windows["energy"], colors["red"],
                               (2 + dimensions["energy"][0] * 0.02 + dimensions["energy"][
                                   0] * 0.96 * self.loc_p / self.line_len,
                                2 + dimensions["energy"][1] * 0.05 + dimensions["energy"][1] * 0.9 * (
                                            1 - (self.energy[self.loc_p] / self.e_pot_initial))), 3)

            # Render slope curve in the sub-window
            slope_line = utils.get_line_to_pygame(self.slope_render_x, np.array(self.slope_render_y) + 20,
                                                  self.line_len, 40,
                                                  dimensions["slope"][0] * 0.96, dimensions["slope"][1] * 0.9,
                                                  dimensions["slope"][0] * 0.02, dimensions["slope"][1] * 0.1)
            pygame.draw.aalines(sub_windows["slope"], color=colors["black"], points=slope_line, closed=False)
            f = interp1d(self.slope_render_x, np.array(self.slope_render_y) + 20, kind='linear')
            pygame.draw.circle(sub_windows["slope"], colors["red"],
                               (2 + dimensions["slope"][0] * 0.02 + dimensions["slope"][
                                   0] * 0.96 * self.loc_p / self.line_len,
                                2 + dimensions["slope"][1] * 0.05 + dimensions["slope"][1] * 0.9 * (
                                            1 - f(self.loc_p) / 40)), 3)

            # Render on-time error curve in the sub-window
            t_error_line = utils.get_line_to_pygame(
                np.concatenate([np.arange(0, len(self.time), self.gap_num), np.array([self.line_len])]),
                15 + (np.array(self.time[::self.gap_num] + [self.time[-1]]) - np.array(
                    self.time_pot[::self.gap_num] + [self.time_pot[-1]])),
                self.line_len, 30,
                dimensions["t_error"][0] * 0.96, dimensions["t_error"][1] * 0.9,
                dimensions["t_error"][0] * 0.02, dimensions["t_error"][1] * 0.05)
            pygame.draw.aalines(sub_windows["t_error"], color=colors["green"], points=t_error_line, closed=False)
            t_error_y = (15 + self.time[self.loc_p] - self.time_pot[self.loc_p]) / 30
            pygame.draw.circle(sub_windows["t_error"], colors["red"],
                               (2 + dimensions["t_error"][0] * 0.02 + dimensions["t_error"][
                                   0] * 0.96 * self.loc_p / self.line_len,
                                2 + dimensions["t_error"][1] * 0.05 + dimensions["t_error"][1] * 0.9 * (1 - t_error_y)),
                               3)

            # Draw sub-windows onto the main window
            self.screen.blit(sub_windows["speed"], (0, 0))
            self.screen.blit(sub_windows["energy"], (dimensions["speed"][0], 0))
            self.screen.blit(sub_windows["slope"], (0, dimensions["speed"][1]))
            self.screen.blit(sub_windows["t_error"], (dimensions["speed"][0], dimensions["energy"][1]))

            # Update display and refresh clock
            self.clock.tick(30)
            pygame.display.flip()

        if self.render_style == 'plt':
            # Render speed curve in the sub-window
            self.l_lb.set_ydata(self.lb_render_y * 3.6)
            self.l_lb.set_xdata(self.lb_render_x)
            self.l_ub.set_ydata(self.ub_render_y * 3.6)
            self.l_ub.set_xdata(self.ub_render_x)
            self.l_mrt.set_ydata(self.mrt_render_y * 3.6)
            self.l_mrt.set_xdata(self.mrt_render_x)
            self.l_v_lim.set_ydata(self.v_lim_render_y)
            self.l_v_lim.set_xdata(self.v_lim_render_x)
            self.l_pot.set_ydata((np.concatenate([self.pot[::self.gap_num], np.array([0])])) * 3.6)
            self.l_pot.set_xdata(np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_speed.set_ydata((np.concatenate([self.speed[::self.gap_num], np.array([0])])) * 3.6)
            self.l_speed.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_pot_loc.set_offsets(np.c_[self.loc_p, self.pot[self.loc_p] * 3.6])
            self.l_speed_loc.set_offsets(np.c_[self.loc_p, self.speed[self.loc_p] * 3.6])

            # ax_gear_action
            self.l_pot_acc.set_ydata((np.concatenate([self.acc_pot[::self.gap_num], np.array([0])])))
            self.l_pot_acc.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_speed_acc.set_ydata((np.concatenate([self.acc[::self.gap_num], np.array([0])])))
            self.l_speed_acc.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_pot_acc_loc.set_offsets(np.c_[self.loc_p, self.acc_pot[self.loc_p]])
            self.l_speed_acc_loc.set_offsets(np.c_[self.loc_p, self.acc[self.loc_p]])

            # ax_energy
            self.l_pot_e.set_ydata(np.concatenate([self.energy_pot[::self.gap_num], np.array([self.energy_pot[-1]])]))
            self.l_pot_e.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_speed_e.set_ydata(np.concatenate([self.energy[::self.gap_num], np.array([self.energy[-1]])]))
            self.l_speed_e.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_pot_e_loc.set_offsets(np.c_[self.loc_p, self.energy_pot[self.loc_p]])
            self.l_speed_e_loc.set_offsets(np.c_[self.loc_p, self.energy[self.loc_p]])

            self.ax_energy.relim()  # Recalculate data limits
            self.ax_energy.autoscale_view()  # Autoscale view
            self.ax_t_error.relim()  # Recalculate data limits
            self.ax_t_error.autoscale_view()  # Autoscale view

            # ax_slope
            self.l_slope.set_ydata(self.slope_render_y)
            self.l_slope.set_xdata(self.slope_render_x)
            f = interp1d(self.slope_render_x, np.array(self.slope_render_y), kind='linear')
            self.l_slope_shift.set_offsets(np.c_[self.slope_render_x, self.slope_render_y])
            self.l_slope_loc.set_offsets(np.c_[self.loc_p, f(self.loc_p)])

            # ax_t_error
            self.l_t_error.set_ydata(
                np.concatenate([self.time[::self.gap_num], np.array([self.time[-1]])]) - np.concatenate(
                    [self.pot_time_initialize[::self.gap_num], np.array([self.pot_time_initialize[-1]])]))
            self.l_t_error.set_xdata(
                np.concatenate([np.arange(0, self.line_len, self.gap_num), np.array([self.line_len])]))
            self.l_t_error_loc.set_offsets(
                np.c_[self.loc_p, self.time[self.loc_p] - self.pot_time_initialize[self.loc_p]])

            plt.draw()
            plt.pause(0.00000000001)  # Pause briefly to observe updates (not needed during actual training)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    train_env = trainEnv()
    tasks = train_env.sample_tasks(5)
    train_env.reset_task({'goal':[0, 0, 900]})

    # 示例数据
    rewards = []

    energy = []
    t_error = []
    jerk = []
    # 模拟训练过程
    for i in range(500):
        done = False

        train_env.is_render = False
        while not done:
            #action = np.array(list(map(float, input("切换点，前功率，后功率：").split(','))))
            action = np.random.rand(3)
            next_state, reward, done, terminal, info = train_env.step(action)
            print(info['info'][3], info['info'][4], info['info'][5])
            if terminal:
                train_env.reset_task(1)
            # train_env.render()
            # train_env.render()
            if done:
                energy.append(info[0])
                t_error.append(info[1])
                jerk.append([info[2]])
                print('回合：', i)
                print("energy:", info[0], ' t_error:', info[1], ' jerk:', info[2])
                print("e_mean:", np.mean(np.array(energy)),
                      ' t_e_mean:', np.mean(np.array(t_error)),
                      ' j_mean:', np.mean(np.array(jerk)))
                print()
                rewards.append(reward)
                train_env.close()

                train_env.reset_task(tasks[2])



    plt.ioff()
    plt.show()

