from gym.envs.registration import register

register(
    id="LSGO-v0",
    entry_point="LSGO_env.TrainContinuous:trainEnv",
    max_episode_steps=2000,
    reward_threshold=1000.0,
)
