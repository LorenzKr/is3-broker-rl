import argparse
import logging
import gym
from ray import rllib
import numpy as np
import ray
from ray.rllib.env.policy_client import PolicyClient
import pandas as pd


# Class to initialize the action and observation space

class Env_config():
    def __init__(self) -> None:
        l_bounds = []
        h_bounds = []
        l_bounds.append(np.array([-np.inf]*24))     #p_grid_imbalance = 0
        h_bounds.append(np.array([-np.inf]*24))                                       
        l_bounds.append(np.array([-np.inf]*24))     #p_customer_prosumption = 0
        h_bounds.append(np.array([-np.inf]*24))                                           
        l_bounds.append(np.array([-np.inf]*24))     #p_wholesale_price = 0
        h_bounds.append(np.array([-np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_cloud_cover = 0
        h_bounds.append(np.array([-np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_temperature = 0
        h_bounds.append(np.array([-np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_wind_speed = 0
        h_bounds.append(np.array([-np.inf]*24))       
        l_bounds.append(np.array([-np.inf]*24))     #p_wind_direction = 0
        h_bounds.append(np.array([-np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     # hour of the start with dummy. 
        h_bounds.append(np.array([-np.inf]*24))
        l_bounds.append(np.array([-np.inf]*7))      # day of the start with dummy
        h_bounds.append(np.array([-np.inf]*7))


        l_bound_total = np.array([])
        for j in l_bounds:
            l_bound_total = np.append(l_bound_total, j)
        r_bound_total = np.array([])
        for j in l_bounds:
            r_bound_total = np.append(r_bound_total, j)


        self.observation_space = gym.spaces.Box(
                    low=l_bound_total,
                    high=r_bound_total,
                    #shape=observation_space_bounds[:, 0].shape,
                )


        self.action_space = gym.spaces.Tuple((gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)), gym.spaces.Discrete(24)))


    def get_gym_spaces(self):
        
        
        return self.observation_space, self.action_space


    def get_rl_config(self):
        config = {
            "env": Env(),
            "observation_space" : self.observation_space,
            "action_space" : self.action_space,
            #"input" : _input,
            "log_level": "DEBUG",
            # === Exploration Settings ===
            # Default exploration behavior, iff `explore`=None is passed into
            # compute_action(s).
            # Set to False for no exploration behavior (e.g., for evaluation).
            #"explore": False,
            "exploration_config": {
                "type": "StochasticSampling",
                "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
            }}
        config.update(
            {
                "num_gpus": 0,
                "model": {"use_lstm": False},
            }
        )
        return config

"""
class Env(gym.Env):
    done = False
    reward = 0
    obs = {}

    def __init__(self, env_config: dict):
        self._log = logging.getLogger(__name__)
        self._log.debug("Creating Env")
        config = Env_config()
        self.observation_space, self.action_space = config.get_gym_spaces()
        
        

    def reset(self):
        self._log.info(f"Reseting env")
        self.reward = 0
        #self.obs = {}
        self.done = False
        return self.obs
  

    def step(self, action):
        obs = self._get_obs()
        self._log.info(f"Taking action: {action}")
        reward = self.reward 
        return obs, reward, self.done, {}

    def _get_obs(self):
        return self.obs

    def _set_obs(self,obs):
        self.obs = obs

"""      
class Env(rllib.env.external_env.ExternalEnv):
    done = False
    reward = 0
    obs = {}
    episode_id = 0

    def __init__(self, env_config: dict):
        self._log = logging.getLogger(__name__)
        self._log.debug("Creating Env")
        config = Env_config()
        self.observation_space, self.action_space = config.get_gym_spaces()
        super().__init__(self.action_space, self.observation_space, max_concurrent=100)
        
   

    def reset(self):
        self._log.info(f"Reseting env")
        self.reward = 0
        #self.obs = {}
        self.done = False
        return self.obs
  

    def start_episode(self, episode_id):
        #obs = self._get_obs()
        self.episode_id = episode_id
        self._log.info(f"Start Episode: {episode_id}")
        return episode_id

    def get_action(self, obs):
        self._log.info(f"Taking action from obs: {obs}")
        super().get_action(self.episode_id, obs)

    def log_action(self, obs):
        self._log.info(f"Taking (log)action from obs: {obs}")
        super().log_action(self.episode_id, obs)

    def log_returns(self, reward):
        self.reward += reward
        super().log_returns(self.episode_id, reward)


    def end_episode(self):
        super().end_episode(self.episode_id)

    def _get_obs(self):
        return self.obs

    def _set_obs(self,obs):
        self.obs = obs