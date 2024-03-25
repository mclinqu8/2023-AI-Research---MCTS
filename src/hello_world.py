import gym
import math
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
# from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import random
import time
from distutils.util import strtobool

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    # TRY NOT TO MODIFY: seeding
    env = gym.vector.SyncVectorEnv(
        [make_env("SpaceInvadersNoFrameskip-v4", 1, 0, False, "")])
    obs = env.reset()
    for global_step in range(1000000):
        action = [env.single_action_space.sample()]
        next_obs, reward, done, _, info = env.step(action)

        # env.unwrapped.restore_full_state() #restores everything completely
        # env.unwrapped.clone_full_state() #clones everything completely
        # env.unwrapped.restore_state() #restores positions
        # env.unwrapped.clone_state() #clones positions


        # TRY NOT TO MODIFY: record rewards for plotting purposes    
        if "final_info" in info.keys() and "episode" in info["final_info"][0].keys():
            print(f"global_step={global_step}, episodic_return={info['final_info'][0]['episode']['r']}")
