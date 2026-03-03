
# Policy Deviation Integral Guided Meta-Reinforcement Learning: Applications to High-speed Train Trajectory Optimization

## Overview
This repository is the official implementation of Policy Deviation Integral Guided Meta-Reinforcement Learning: Applications to High-speed Train Trajectory Optimization, currently under review at IEEE Transactions on Intelligent Transportation Systems (T-ITS), 2026. 
![image](https://github.com/Tongzh-SEU/PDIMRL-LSGO/blob/main/Images/Framework.png)
This repository contains two key components:
1. **LSGO Environment**: A custom Gym environment designed to address the high-speed rail （HSR）Train Trajectory Optimization (TTO) problem using a novel reconceptualization approach (It can be used as a TTO environment for other DRL methods).
2. **PDIMRL Algorithm**: A novel first-order gradient-based meta-reinforcement learning (meta-RL) algorithm for HSR operation tasks, combining Proximal Policy Optimization (PPO) with Policy Deviation Integre (PDI) and Reptile meta-learning.

## Requirements

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
## Files
- train.py: Runs PDIMRL training in LSGO environment
- PDIMRL.py: PDIMRL algorithm implementation
- LSGO_env: LSGO-centirc MDP implementation
- /data: Custom environment parameters and cached data
- /get_data.py: Precompute environment cache from custom parameters
- /TrainContinuous.py: MDP Implementation for train trajectory optimization

## Configure & Train

- **Customize LSGO-centric MDP parameters**<br>
  Edit files in `LSGO_env/data/parameter/`:
  - `slope` (track gradient)
    e.g. [0m,1520m) slope is 0‰; [1520m,5000m) slope is 2‰, etc.
    | station | seg_slope | slope |
    |---|---|---|
    | 0 | 0 | 0 |
    | 0 | 1520 | 2 |
    | 0 | 5000 | 0 |
  - `speed_limit` (speed constraints)
    e.g. [0m,1000m) v lim is 80km/h; [1000m,10000m) v lim is 300km/h, etc.
    | station | seg_v_lim | v_lim |
    |---|---|---|
    | 0 | 0 | 80 |
    | 0 | 1000 | 300 |
    | 0 | 10000 | 0 |
  - `time` (task settings)
    e.g. task 1 is 800s, task 2 is 900s, etc.
    | station | plan_time|
    |---|---|
    | 0 | 800 |
    | 0 | 900 |
  - `train` (train dynamics)

- **Generate environment cache data**<br>
  This generates cached data for the current parameter configuration in a new env parameters, and executed only on the first run after parameter changes.
  ```bash
  python LSGO_env/get_data.py

- **Environment render settings**<br>
  Edit code in ‘/TrainContinuous.py’
    - `self.is_render(bool)`: enable visualization.
    - `self.render_gap(int)`: render refresh gap (in meters).
    - `self.gap_num (int)`: Rendering precision.  

- **Configure PDIMRL and meta-tasks**<br>
  Configure PDIMRL hyperparameters and meta-task definitions in train.py.<br>
  meta-task definitions eg. tasks[0] is station 0, plan_time index 0, and plan time legend is 800, etc.<br>
  `tasks = [{'goal': [0, 0, 800]}, {'goal': [0, 1, 900]}]`

- **Runs PDIMRL training in LSGO environment**<br>
  ```bash
  python train.py
