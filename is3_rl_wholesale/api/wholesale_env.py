import argparse
import logging
import gym
from ray import rllib
import numpy as np
import ray
from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from ray.rllib.agents import impala


# Class to initialize the action and observation space

class Env_config():
    def __init__(self) -> None:
        l_bounds = []
        h_bounds = []
        l_bounds.append(np.array([-np.inf]*24))     #p_grid_imbalance = 0
        h_bounds.append(np.array([np.inf]*24))                                       
        l_bounds.append(np.array([-np.inf]*24))     #p_customer_prosumption = 0
        h_bounds.append(np.array([np.inf]*24))                                           
        l_bounds.append(np.array([-np.inf]*24))     #p_wholesale_price = 0
        h_bounds.append(np.array([np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_cloud_cover = 0
        h_bounds.append(np.array([np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_temperature = 0
        h_bounds.append(np.array([np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     #p_wind_speed = 0
        h_bounds.append(np.array([np.inf]*24))       
        l_bounds.append(np.array([-np.inf]*24))     #p_wind_direction = 0
        h_bounds.append(np.array([np.inf]*24))     
        l_bounds.append(np.array([-np.inf]*24))     # hour of the start with dummy. 
        h_bounds.append(np.array([np.inf]*24))
        l_bounds.append(np.array([-np.inf]*7))      # day of the start with dummy
        h_bounds.append(np.array([np.inf]*7))


        l_bound_total = np.array([])
        for j in l_bounds:
            l_bound_total = np.append(l_bound_total, j)
        r_bound_total = np.array([])
        for j in h_bounds:
            r_bound_total = np.append(r_bound_total, j)


        self.observation_space = gym.spaces.Box(
                    low=np.ravel(l_bound_total),
                    high=np.ravel(r_bound_total),
                    dtype=np.float32
                    #shape=observation_space_bounds[:, 0].shape,
                )


        #self.action_space = gym.spaces.Tuple((gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)), gym.spaces.Discrete(24)))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    def get_gym_spaces(self):
        
        
        return self.observation_space, self.action_space


    def get_rl_config(self):
        config = {
            #"env": Env({}),
            #"observation_space" : self.observation_space,
            #"action_space" : self.action_space,
            #"input" : _input,
            "log_level": "DEBUG",
            
            # === Exploration Settings ===
            # Default exploration behavior, iff `explore`=None is passed into
            # compute_action(s).
            # Set to False for no exploration behavior (e.g., for evaluation).
            #"explore": False,
           # "exploration_config": {
            #    "type": "StochasticSampling",
           #     "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
           # }
           }
        """
        config.update(
            {
                "num_gpus": 0,
                "model": {"use_lstm": False},
            }
            )
            """
        
        config.update({
            
            # V-trace params (see vtrace_tf/torch.py).
            "vtrace": True,
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            # If True, drop the last timestep for the vtrace calculations, such that
            # all data goes into the calculations as [B x T-1] (+ the bootstrap value).
            # This is the default and legacy RLlib behavior, however, could potentially
            # have a destabilizing effect on learning, especially in sparse reward
            # or reward-at-goal environments.
            # False for not dropping the last timestep.
            "vtrace_drop_last_ts": True,
            # System params.
            #
            # == Overview of data flow in IMPALA ==
            # 1. Policy evaluation in parallel across `num_workers` actors produces
            #    batches of size `rollout_fragment_length * num_envs_per_worker`.
            # 2. If enabled, the replay buffer stores and produces batches of size
            #    `rollout_fragment_length * num_envs_per_worker`.
            # 3. If enabled, the minibatch ring buffer stores and replays batches of
            #    size `train_batch_size` up to `num_sgd_iter` times per batch.
            # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
            #    on batches of size `train_batch_size`.
            #
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "min_time_s_per_reporting": 10,
            "num_workers": 2,
            # Number of GPUs the learner should use.
            "num_gpus": 0,
            # For each stack of multi-GPU towers, how many slots should we reserve for
            # parallel data loading? Set this to >1 to load data into GPUs in
            # parallel. This will increase GPU memory usage proportionally with the
            # number of stacks.
            # Example:
            # 2 GPUs and `num_multi_gpu_tower_stacks=3`:
            # - One tower stack consists of 2 GPUs, each with a copy of the
            #   model/graph.
            # - Each of the stacks will create 3 slots for batch data on each of its
            #   GPUs, increasing memory requirements on each GPU by 3x.
            # - This enables us to preload data into these stacks while another stack
            #   is performing gradient calculations.
            "num_multi_gpu_tower_stacks": 0,
            # How many train batches should be retained for minibatching. This conf
            # only has an effect if `num_sgd_iter > 1`.
            "minibatch_buffer_size": 1,
            # Number of passes to make over each train batch.
            "num_sgd_iter": 1,
            # Set >0 to enable experience replay. Saved samples will be replayed with
            # a p:1 proportion to new data samples.
            "replay_proportion": 0.0,
            # Number of sample batches to store for replay. The number of transitions
            # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
            "replay_buffer_num_slots": 0,
            # Max queue size for train batches feeding into the learner.
            "learner_queue_size": 16,
            # Wait for train batches to be available in minibatch buffer queue
            # this many seconds. This may need to be increased e.g. when training
            # with a slow environment.
            "learner_queue_timeout": 300,
            # Level of queuing for sampling.
            "max_sample_requests_in_flight_per_worker": 2,
            # Max number of workers to broadcast one set of weights to.
            "broadcast_interval": 1,
            # Use n (`num_aggregation_workers`) extra Actors for multi-level
            # aggregation of the data produced by the m RolloutWorkers
            # (`num_workers`). Note that n should be much smaller than m.
            # This can make sense if ingesting >2GB/s of samples, or if
            # the data requires decompression.
            "num_aggregation_workers": 0,

            # Learning params.
            "grad_clip": 40.0,
            # Either "adam" or "rmsprop".
            "opt_type": "adam",
            "lr": 0.0005,
            "lr_schedule": None,
            # `opt_type=rmsprop` settings.
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            # Balancing the three losses.
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,
            # Set this to true to have two separate optimizers optimize the policy-
            # and value networks.
            "_separate_vf_optimizer": False,
            # If _separate_vf_optimizer is True, define separate learning rate
            # for the value network.
            "_lr_vf": 0.0005,

            # Callback for APPO to use to update KL, target network periodically.
            # The input to the callback is the learner fetches dict.
            "after_train_step": None,

        
        })
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
    #obs = {}
    #episode_id = 0
    timeslot = set()
    backup_obs = np.array([-5.51846826e03 ,-6.34140234e03, -7.33104980e03, -7.19497510e03, -7.32590234e03, -7.83731738e03,
        -8.20035547e03, -1.28710791e03,
        1.02296250e04 , 1.48304512e04 , 1.27051992e04,  5.63840186e03,
        1.11968408e03 ,-3.37990771e03 ,-7.34375439e03, -1.06434189e04,
        -1.89569590e04, -3.22251133e04, -3.97013164e04, -4.18393047e04,
        -4.17268047e04, -4.23558828e04, -4.08356289e04, -4.12660234e04,
        -5.37647070e04, -5.21676719e04, -4.70058398e04, -3.83295156e04,
        -3.35923906e04, -3.43150664e04, -3.68412031e04, -3.40671484e04,
        -2.79747695e04, -2.23473574e04, -2.52260312e04, -3.94131055e04,
        -5.00172461e04, -5.24158633e04, -5.17868477e04, -5.22881719e04,
        -5.74263320e04, -6.06062617e04, -6.20960273e04, -6.24544492e04,
        -6.26435195e04, -6.29509961e04, -6.28256641e04, -6.28178555e04,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 0.00000000e00 , 0.00000000e00,  0.00000000e00,
        0.00000000e00 , 5.83000000e01 , 9.45000000e01,  6.97000000e01,
        9.57000000e01 , 9.30000000e01 , 4.54000000e01,  7.10000000e02,
        5.00000000e02 , 2.70000000e01 , 6.46000000e01,  9.24000000e01,
        8.49000000e01 , 5.59000000e01 , 2.94000000e01,  0.00000000e00,
        0.00000000e00 , 1.50000000e02 , 1.03000000e01,  1.00000000e00,
        4.67000000e01 , 0.00000000e00 , 9.54000000e01,  8.39000000e01,
        1.33000000e01 , 1.01000000e01 , 1.06000000e01,  1.04000000e01,
        1.17000000e01 , 1.13000000e01 , 1.19000000e01,  1.18000000e01,
        1.26000000e01 , 1.36000000e01 , 1.46000000e01,  1.51000000e01,
        1.49000000e01 , 1.54000000e01 , 1.52000000e01,  1.59000000e01,
        1.47000000e01 , 1.40000000e01 , 1.28000000e01,  1.20000000e01,
        1.32000000e01 , 1.44000000e01 , 9.70000000e00,  1.00000000e01,
        6.07000000e00 , 5.95000000e00 , 6.90000000e00,  6.92000000e00,
        6.85000000e00 , 5.66000000e00 , 6.40000000e00,  6.79000000e00,
        7.05000000e00 , 7.84000000e00 , 7.98000000e00,  8.03000000e00,
        8.04000000e00 , 6.75000000e00 , 6.85000000e00,  6.13000000e00,
        5.26000000e00 , 5.50000000e00 , 6.36000000e00,  3.52000000e00,
        5.98000000e00 , 6.07000000e00 , 6.27000000e00,  7.37000000e00, 1.03000000e01,  1.00000000e00,
        4.67000000e01 , 0.00000000e00 , 9.54000000e01,  8.39000000e01,
        1.33000000e01 , 1.01000000e01 , 1.06000000e01,  1.04000000e01,
        1.17000000e01 , 1.13000000e01 , 1.19000000e01,  1.18000000e01,
        1.26000000e01 , 1.36000000e01 , 1.46000000e01,  1.51000000e01,
        1.49000000e01 , 1.54000000e01 , 1.52000000e01,  1.59000000e01,
        1.47000000e01 , 1.40000000e01 , 1.28000000e01,  1.20000000e01,
        1.32000000e01 , 1.44000000e01 , 9.70000000e00,  1.00000000e01,
        6.07000000e00 , 5.95000000e00 , 6.90000000e00,  6.92000000e00,
        6.85000000e00 , 5.66000000e00 , 6.40000000e00,  6.79000000e00,
        7.05000000e00 , 7.84000000e00 , 7.98000000e00,  8.03000000e00,
        8.04000000e00 , 6.75000000e00 , 6.85000000e00,  6.13000000e00,
        5.26000000e00 , 5.50000000e00 , 6.36000000e00,  3.52000000e00,
        5.98000000e00 , 6.07000000e00 , 6.27000000e00,  7.37000000e00,  7.37000000e00],dtype=np.float32)

    def __init__(self, env_config: dict):
        self._log = logging.getLogger(__name__)
        self._log.debug("Creating Env")
        #self.backup_obs = self.obs
        config = Env_config()
        self.observation_space, self.action_space = config.get_gym_spaces()
        #self._log.info(np.shape(self.obs), self.obs)
        super().__init__(self.action_space, self.observation_space, max_concurrent=100)
        

    def run(self):
        self._log.info("Running run once.")
        #self.reset()
        timeslot = 0
        self.episode_id = self.start_episode()
        self._log.info("Running run once.")
        for i in range(1):
            self._log.info("Running run in while.")
            if timeslot not in self.timeslot:
                self._log.info("Running run once.")
                #obs = self._get_obs()
                self._log.info("Running run once.")
                self.action = self.get_action(episode_id= self.episode_id,observation = self.backup_obs)
                self._log.info("Running run once.")
                self._log.info(f"Action in while {self.action}")
                
"""
    def reset(self):
        self._log.info(f"Reseting env")
        self.reward = 0
        self.obs = self.backup_obs
        self.done = False
        return self.obs
  

    def start_episode(self):
        self._log.info("Running run once.")
        #obs = self._get_obs()
        #episode_id = "0"
        #self.episode_id = episode_id
        
        self.episode_id = super().start_episode()
        self._log.info(f"Start Episode: {self.episode_id}")
        return self.episode_id

    def get_action(self, obs):
        self._log.info("Running get_action once.")
        try:
            action = super().get_action(self.episode_id, obs)
        except Exception as e:
            self._log.error(e)
        self._log.info(f"Taking action: {action}")
        return action

    def log_action(self, obs, action):
        self._log.info(f"Logging action from obs: {obs} with action {action}")
        super().log_action(self.episode_id, obs, action)

    def log_returns(self, reward, episode_id):
        self._log.info("Running run once.")
        self.reward += reward
        super().log_returns(self.episode_id, reward)


    def end_episode(self):
        self._log.info("Running run once.")
        super().end_episode(self.episode_id)

    


    def _get_obs(self):
        return self.obs

    def _set_obs(self,obs):
        self.obs = obs
"""