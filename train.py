import os
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("QtAgg")   # 或 "TkAgg"
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import pandas as pd
import LSGO_env.TrainContinuous
import gym
import t_utils
import copy
from collections import OrderedDict
from PDIMRL import PDIMRL






################################### Training ###################################
def train():
    global temp_params
    print("============================================================================================")




    ####### initialize environment hyperparameters ######
    env_name = "LSGO-v0"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 14  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 20  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.1  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.02  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(1e4)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PDIMRL hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 2  # update policy for K epochs in one PDIMRL update
    N_task = 15  #inner loop num
    M_update = 4
    T_iterations = 10
    eps_clip = 0.2  # clip parameter for PDIMRL
    gamma = 0.99  # discount factor

    reptile_policy_lr = 1
    reptile_t_i = (1*0.8)/T_iterations

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    LSGO_env = gym.make(env_name)

    # state space dimension
    state_dim = LSGO_env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = LSGO_env.action_space.shape[0]
    else:
        action_dim = LSGO_env.action_space.n


    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PDIMRL_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PDIMRL_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PDIMRL update frequency : " + str(update_timestep) + " timesteps")
    print("PDIMRL K epochs : ", K_epochs)
    print("PDIMRL epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        LSGO_env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    tasks = [{'goal': [0, 6, 1100]},
             {'goal': [0, 4, 1000]},
             {'goal': [0, 8, 1200]},
             {'goal': [0, 0, 800]},
             {'goal': [0, 7, 1150]},
             {'goal': [0, 3, 950]},
             {'goal': [0, 1, 850]},
             {'goal': [0, 5, 1050]},
             {'goal': [0, 2, 900]},
             {'goal': [0, 9, 1250]},
             ]
    tasks_pot = LSGO_env.n_pot
    n_pot_actions = []
    n_pot_e_thr = []
    for i in range(len(tasks_pot)):
        n_pot_e_thr.append([])
        n_pot_actions.append([])
        for j in range(len(tasks_pot[i])):
            actions = t_utils.get_line_action(tasks_pot[i][j], LSGO_env.n_slope_x[i], LSGO_env.n_slope_y[i], LSGO_env.param)
            n_pot_e_thr[i].append(
                t_utils.get_line_energy(tasks_pot[i][j], LSGO_env.n_slope_x[i], LSGO_env.n_slope_y[i], LSGO_env.param))
            actions = t_utils.compress_array(actions)
            n_pot_actions[i].append(actions)

    # initialize a PDIMRL agent
    PDIMRL_agent = PDIMRL(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std, tasks)
    # PDIMRL_agent.load('PDIMRL_TrainContinuous-v2_0_0.pth')

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # printing and logging variables
    print_running_reward = 0
    print_done_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    energy = []
    real_t = []
    jerk = []

    e_r = []
    t_r = []
    j_r = []
    all_R = []

    e_thr = []
    plan_t = []

    r_action_std = []
    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(5, 2, figure=fig, width_ratios=[10, 5], height_ratios=[4, 1, 4, 1, 4])
    ax_reward = fig.add_subplot(gs[0:5, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_t_error = fig.add_subplot(gs[2, 1])
    ax_jerk = fig.add_subplot(gs[4, 1])

    ax_reward.set_title('Reward')
    ax_energy.set_title('Energy')
    ax_t_error.set_title('T Error')
    ax_jerk.set_title('Action Std')

    ax_reward.grid()
    ax_energy.grid()
    ax_t_error.grid()
    ax_jerk.grid()

    energy_out = []
    t_errror_out = []

    line_all_R, = ax_reward.plot(all_R, label='all_reward')
    #line_e_r, = ax_reward.plot(e_r, label='E_reward')
    #line_t_r, = ax_reward.plot(t_r, label='T_reward')
    #line_j_r, = ax_reward.plot(j_r, label='J_reward')

    line_e_thr, = ax_energy.plot(e_thr, label='E_thr')
    line_energy, = ax_energy.plot(energy, label='energy')

    line_t_plan, = ax_t_error.plot(real_t, label='t_plan')
    line_t_real, = ax_t_error.plot(real_t, label='t_real')


    # line_jerk, = ax_jerk.plot(jerk, label='jerk')
    line_std, = ax_jerk.plot(r_action_std, label='action_std')

    reward_mean = 0
    is_first = True
    epoach = 0
    # training loop
    for t in range(T_iterations):
        print(f'meta_iteration {t + 1}/{T_iterations}')
        # tasks = LSGO_env.sample_tasks(10)
        tasks = [{'goal': [0, 6, 1100]},
                 {'goal': [0, 4, 1000]},
                 {'goal': [0, 8, 1200]},
                 {'goal': [0, 0, 800]},
                 {'goal': [0, 7, 1150]},
                 {'goal': [0, 3, 950]},
                 {'goal': [0, 1, 850]},
                 {'goal': [0, 5, 1050]},
                 {'goal': [0, 2, 900]},
                 {'goal': [0, 9, 1250]},
                 ]

        for task_i in range(len(tasks)):
            print(f'task {task_i + 1}/{len(tasks)} {tasks[task_i]["goal"]}')
            temp_params = copy.deepcopy(OrderedDict(PDIMRL_agent.policy.named_parameters()))
            task = tasks[task_i]
            # PDIMRL_agent.c_buffer.set_positive_task(task)
            LSGO_env.reset_task(task)
            max_ep_len = LSGO_env.step_len
            PDIMRL_agent.policy.load_state_dict(temp_params)
            PDIMRL_agent.policy_old.load_state_dict(temp_params)  # load initial params for our model
            # load initial params for our model
            PDIMRL_agent.init_optimizers()

            state = LSGO_env.reset()
            while True:
                if t == 0 and task_i == 0:
                    break
                PDIMRL_agent.policy.eval()
                PDIMRL_agent.policy_old.eval()
                action = PDIMRL_agent.select_action(state)
                for i in range(len(action)):
                    action[i] = min(max(action[i], 0), 1)
                state, reward, done, _, info = LSGO_env.step(action)


                if done:
                    agent_action = t_utils.get_line_action(info['speed'], LSGO_env.n_slope_x[task['goal'][0]],
                                                           LSGO_env.n_slope_y[task['goal'][0]], LSGO_env.param)
                    agent_actions = t_utils.compress_array(agent_action)

                    integral_agent_actions = t_utils.integrate_compressed(agent_actions)
                    integral_pot_actions = t_utils.integrate_compressed(n_pot_actions[task['goal'][0]][task['goal'][1]])

                    ratio_factor = integral_pot_actions/integral_agent_actions



                    PDIMRL_agent.policy_deviation_integral(factor=1 / ratio_factor, lr = reptile_policy_lr)

                    LSGO_env.reset()
                    PDIMRL_agent.policy.train()
                    PDIMRL_agent.policy_old.train()
                    break

            for n in range(N_task):
                for m in range(M_update):
                    state = LSGO_env.reset()
                    current_ep_reward = 0
                    current_ep_done_reward = 0
                    for t in range(1, max_ep_len + 1):
                        # select action with policy
                        action = PDIMRL_agent.select_action(state)
                        for i in range(len(action)):
                            action[i] = min(max(action[i], 0), 1)

                        state, reward, done, ter, info = LSGO_env.step(action)


                        if info['info'][0] < n_pot_e_thr[task['goal'][0]][task['goal'][1]] and abs(info['info'][1]) < 1:
                            n_pot_e_thr[task['goal'][0]][task['goal'][1]] = info['info'][0]
                            actions = t_utils.get_line_action(info['speed'], LSGO_env.n_slope_x[task['goal'][0]], LSGO_env.n_slope_y[task['goal'][0]], LSGO_env.param)
                            actions = t_utils.compress_array(actions)
                            n_pot_actions[task['goal'][0]][task['goal'][1]] = actions


                        if info['info'][0] < info['info'][6] and info['info'][1] < 1:
                            energy_out.append(info['info'][0])
                            t_errror_out.append(info['info'][1])



                        if len(energy_out) % 800 == 0 and len(energy_out) != 0:
                            data = {
                                'energy_out': energy_out,
                                't_errror_out': t_errror_out
                            }
                            df = pd.DataFrame(data)

                            df.to_excel('output.xlsx', index=False)
                        if done:
                            epoach += 1


                            if is_first:
                                reward_mean = max(reward, -30)
                                is_first = False
                            else:
                                reward_mean = reward_mean * 0.99 + max(reward, -30) * 0.01

                            reward_action_std = min(max(0.01, 1 * action_std * max((6 - reward_mean) / 24, 0)), 0.5)
                            PDIMRL_agent.set_action_std(max(reptile_policy_lr*0.15,0.01))

                            r_action_std.append(reward_action_std)

                            energy.append(info['info'][0])
                            real_t.append(info['info'][1]+LSGO_env.runtime)
                            jerk.append(info['info'][2])
                            e_r.append(info['info'][3])
                            t_r.append(info['info'][4])
                            j_r.append(info['info'][5])
                            all_R.append(reward)
                            e_thr.append(info['info'][6])
                            plan_t.append(LSGO_env.runtime)


                            line_all_R.set_ydata(all_R[-8000:])
                            line_all_R.set_xdata(range(len(all_R[-8000:])))


                            line_e_thr.set_ydata(e_thr[-8000:])
                            line_e_thr.set_xdata(range(len(all_R[-8000:])))
                            line_energy.set_ydata(energy[-8000:])
                            line_energy.set_xdata(range(len(all_R[-8000:])))

                            line_t_plan.set_ydata(plan_t[-8000:])
                            line_t_plan.set_xdata(range(len(all_R[-8000:])))
                            line_t_real.set_ydata(real_t[-8000:])
                            line_t_real.set_xdata(range(len(all_R[-8000:])))

                            line_std.set_ydata(r_action_std[-8000:])
                            line_std.set_xdata(range(len(all_R[-8000:])))

                            ax_reward.relim()
                            ax_reward.autoscale_view()
                            ax_energy.relim()
                            ax_energy.autoscale_view()
                            ax_t_error.relim()
                            ax_t_error.autoscale_view()
                            ax_jerk.relim()
                            ax_jerk.autoscale_view()

                            ax_reward.legend()
                            ax_energy.legend()
                            ax_t_error.legend()
                            ax_jerk.legend()

                            plt.draw()
                            plt.pause(0.00000000001)



                        # saving reward and is_terminals
                        PDIMRL_agent.buffer.rewards.append(reward)
                        PDIMRL_agent.buffer.is_terminals.append(done)
                        PDIMRL_agent.buffer.speeds.append(info['speed'])


                        time_step += 1
                        current_ep_reward += reward
                        if done:
                            current_ep_done_reward += reward
                            break





                        # printing average reward
                        if time_step % print_freq == 0:
                            # print average reward till last episode
                            print_avg_reward = print_running_reward / print_running_episodes
                            print_avg_reward = round(print_avg_reward, 2)
                            print_done_reward = print_done_reward / print_running_episodes
                            print_done_reward = round(print_done_reward, 2)
                            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Done Reward : {}".format(i_episode, time_step,
                                                                                                                          print_avg_reward, print_done_reward))
                            print_done_reward = 0
                            print_running_reward = 0
                            print_running_episodes = 0

                        # save model weights
                        if time_step % save_model_freq == 0:
                            print("--------------------------------------------------------------------------------------------")
                            print("saving model at : " + checkpoint_path)
                            PDIMRL_agent.save(checkpoint_path)
                            print("model saved")
                            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                            print("--------------------------------------------------------------------------------------------")

                        # break; if the episode is over
                        if done:
                            break

                    print_done_reward += current_ep_done_reward
                    print_running_reward += current_ep_reward
                    print_running_episodes += 1

                    log_running_reward += current_ep_reward
                    log_running_episodes += 1
                    i_episode += 1

                # update PDIMRL agent
                print(f'PDIMRL inner update {n+1}/{N_task}')
                PDIMRL_agent.update()

            print(f'PDIMRL outer update {task_i+1}/{len(tasks)}')
            target_policy = OrderedDict(PDIMRL_agent.policy.named_parameters())  # get target params from this training
            temp_params = PDIMRL_agent.update_init_params(target_policy, temp_params, reptile_policy_lr*0.1)  # update our params from this task

            PDIMRL_agent.save(checkpoint_path)
            LSGO_env.close()
        PDIMRL_agent.policy.load_state_dict(temp_params)  # update our network
        PDIMRL_agent.policy_old.load_state_dict(temp_params)  # update our network
        PDIMRL_agent.save(checkpoint_path)
        reptile_policy_lr = reptile_policy_lr - reptile_t_i

    LSGO_env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()







