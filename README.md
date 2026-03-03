
# Policy Deviation Integral Guided Meta-Reinforcement Learning: Applications to High-speed Train Trajectory Optimization

## Overview
This repository is the official implementation of Policy Deviation Integral Guided Meta-Reinforcement Learning: Applications to High-speed Train Trajectory Optimization, currently under review at IEEE Transactions on Intelligent Transportation Systems (T-ITS), 2025. This repository contains two key components:
1. **LSGO Environment**: A custom Gym environment designed to address the high-speed rail （HSR）Train Trajectory Optimization (TTO) problem using a novel reconceptualization approach.
2. **PDIMRL Algorithm**: A novel first-order gradient-based meta-reinforcement learning (meta-RL) algorithm for HSR operation tasks, combining Proximal Policy Optimization (PPO) with Policy Deviation Integre (PDI) and Reptile meta-learning.

![image](https://github.com/Tongzh-SEU/PDIMRL-LSGO/blob/main/Images/Framework.png)

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
# Files
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
  - `speed_limit` (speed constraints)
  - `time` (task settings)
  - `train` (train dynamics)

- **Generate environment cache data**<br>
  This generates cached data for the current parameter configuration.  
  ```bash
  python LSGO_env/get_data.py

- **Environment render settings**<br>
  Edit code in ‘/TrainContinuous.py’
    - `self.is_render = True`: enable visualization.
    - `self.render_gap = self.line_len`: render refresh gap (in meters).
    - `self.gap_num = 100`: Rendering precision.  

- **Configure PDIMRL and meta-tasks**<br>
  Configure PDIMRL hyperparameters and meta-task definitions in train.py.

- **Runs PDIMRL training in LSGO environment**<br>
  ```bash
  python train.py


  
