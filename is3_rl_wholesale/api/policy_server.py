import logging
from pprint import PrettyPrinter

from ray import serve
from starlette.requests import Request
import time
import ray
import gym
from ray import rllib
import numpy as np
import pickle
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks

_log = logging.getLogger(__name__)
if __name__ == "__main__":
    SERVER_ADDRESS = "localhost"
    SERVER_BASE_PORT = 9000  # + worker-idx - 1

    observation_space_bounds = np.array(
            [
                [-np.inf, np.inf],    #p_grid_imbalance = 0
                [-np.inf, np.inf],                #p_customer_prosumption = 0
                [-np.inf, np.inf],                #p_wholesale_price = 0
                [-np.inf, np.inf],                #p_cloud_cover = 0
                [-np.inf, np.inf],                #p_temperature = 0
                [-np.inf, np.inf],                #p_wind_speed = 0
                [-np.inf, np.inf],                #p_wind_direction = 0
                [0, 7],                #p_day_of_week = 0
                [-np.inf, np.inf],                #p_hour_of_day = 0
                [-np.inf, np.inf],    #competing_broker_identities = []  # A list of competing brokers
                    #cleared_trade = []  # The cleared trade that the message is sending.]
            ]
            )

    observation_space = gym.spaces.Box(
                low=observation_space_bounds[:, 0],
                high=observation_space_bounds[:, 1],
                shape=observation_space_bounds[:, 0].shape,
            )

    action_space_bounds = np.array(
                [
                    [-np.inf, np.inf],  # kWh
                    [-np.inf, np.inf],  # price
                    [0,23] # timeslot
                ]
            )
    action_space = gym.spaces.Box(
                low=action_space_bounds[:, 0],
                high=action_space_bounds[:, 1]
            )


    ray.init()
    serve.start(detached=True)
    def _input(ioctx):
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                9000 + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None


    config = {
        "env": None,
        "observation_space" : observation_space,
        "action_space" : action_space,
        "input" : _input,
        "log_level": "DEBUG",
        "model": {},
        # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    #"explore": False,
    "exploration_config": {
   "type": "StochasticSampling",
   "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
},"input_evaluation": []}
    config.update(
                {
                    "num_gpus": 0,
                    "model": {"use_lstm": True},
                }
            )
   
    trainer_cls = get_trainer_class("PPO")
    trainer = trainer_cls(config=config)

    ts = 0

    while True:
        trainer.train()
    #while True:
    #    time.sleep(5)
    #policy_server = policy_server_input(,port=8000)
    #policy_server