
# LSGO Environment & PDIMRL Algorithm

## Overview

![image](https://github.com/Tongzh-SEU/PDIMRL-LSGO/blob/main/Images/Framework.pdf)

This repository contains two key components:

1. **LSGO Environment**: A custom Gym environment designed to address the high-speed rail （HSR）Train Trajectory Optimization (TTO) problem using a novel reconceptualization approach.
2. **PDIMRL Algorithm**: A novel first-order gradient-based meta-reinforcement learning (meta-RL) algorithm for HSR operation tasks, combining Proximal Policy Optimization (PPO) with Policy Deviation Integre (PDI) and Reptile meta-learning.



## Dependent Environment

Ensure that the following dependencies are installed in your Python environment before running the code:

- `python: 3.8.16`
- `numpy: 1.24.2`
- `torch: 2.0.0`
- `gym: 0.26.2`
- `matplotlib: 3.7.1`
- `pandas: 1.5.3`
- `scipy: 1.10.1`
- `pygame: 2.3.0`

You can install the required packages using the following command:

```bash
pip install numpy==1.24.2 torch==2.0.0 gym==0.26.2 matplotlib==3.7.1 pandas==1.5.3 scipy==1.10.1 pygame==2.3.0
```

## LSGO Environment

### Overview

The **LSGO Environment** redefines the traditional TTO (Train Trajectory Optimization) issue. In this environment, instead of directly seeking the optimal solution, the agent refines a suboptimal PMP. The action space is compressed from a unidimensional scale into a compact three-dimensional domain, encapsulating both retrospective and prospective state horizons. The reward function is designed for on-time evaluation, avoiding delayed feedback.


### Structure

- **Action Space**: A three-dimensional domain.
- **State Space**: Contains both retrospective and prospective elements.
- **Reward Function**: Designed for immediate feedback, evaluated on-time to avoid delays.

### Files & Directories

- The environment code is located in the `./LSGO Env/` directory.
- Environment parameter data is stored in `./LSGO Env/LSGO/data/`.
- You can modify the files under `./LSGO Env/LSGO/data/parameter/` to create different meta-environment tasks, which contain gradients, speed limits, and train parameters as well as scheduling tasks.
- The data generation script is available at `./LSGO Env/LSGO/get_data.py`, which allows you to generate new environment data.
  
### Registration

To use the environment, you need to place the files in `LSGO Env/` under the `./gym.envs/` directory and register the environment in `./gym.envs/__init__.py`:

### Render Modes

The environment supports two render modes:

1. **Plt (matplotlib)**: For visualizing data plots.
2. **Pygame**: For graphical interaction.(Recommended)

```bash
self.render_style = 'plt'
self.is_render = True
# Display calculation points separated by render
self.render_gap = 1000
# Fill to render precision
self.gap_num = 50
```

You can switch between these modes based on your needs during testing and debugging.



---

## PDIMRL Algorithm

### Overview

The **PDIMRL Algorithm** (Policy Deviation Integre Meta-Reinforcement Learning) is a first-order gradient-based meta-RL algorithm that is designed for efficient adaptation across a distribution of tasks with complex dynamic constraints, specifically in high-speed rail (HSR) operations.


Key components:
- **Outer Meta-Learning Loop**: Integrates Policy Deviation Integre (PDI) with Reptile meta-learning to initialize neural network parameters for fast adaptation.
- **Inner DRL Loop**: Utilizes Proximal Policy Optimization (PPO) to iteratively refine policy parameters.
- **Goal**: Accelerates learning by initializing neural network models efficiently, enabling adaptation with limited training data.

### Structure

- **Main Program**: Located in `PDIMRL algorithm/train.py`.
- **PPO and PDI**: Found in `PDIMRL algorithm/PPO.py`.
  - The `policy_deviation_integral` method implements the PDI algorithm.
  - The `update_init_params` method in `train.py` implements the Reptile meta-learning update.
- **Control Loop**: The control loop and interactions with the LSGO environment are implemented in `train.py`.

### Running the PDIMRL Algorithm

To run the algorithm, execute the following:

```bash
python PDIMRL algorithm/train.py
```

This will train the agent in the LSGO environment using the PDIMRL meta-learning approach.

---

## How to Modify and Customize

1. **Generate Data**: Use the `get_data.py` script under `./LSGO Env/LSGO/` to generate new environment parameter data.
2. **Modify Meta-Tasks**: Customize the environment by editing files under `./LSGO Env/LSGO/data/parameter/` to create different meta-environment tasks.
3. **Train the Agent**: Train the agent by running the main training script at `PDIMRL algorithm/train.py`.

### Example Code Snippets

Here is an example of how to register and use the LSGO environment:

```python
import gym

env = gym.make('LSGO-v0')
observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode='plt')
```

---

## Conclusion

This repository provides a framework for solving dynamic, meta-learning based optimization tasks in complex environments using both the LSGO environment and the PDIMRL algorithm. By modifying environment parameters and training the model using meta-RL techniques, you can explore various optimization challenges and tasks efficiently.

