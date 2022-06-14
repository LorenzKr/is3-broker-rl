import json
import logging

from requests import request
from conf import setup_logging
from ray import serve
from starlette.requests import Request
import time
import ray
import gym
from ray import rllib
import numpy as np
import pickle
import sys
from fastapi import FastAPI
from is3_rl_wholesale.api.wholesale_env import Env, Env_config
from ray.rllib.agents.registry import get_trainer_class
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

app = FastAPI()
ray.init(address="auto", namespace="serve", ignore_reinit_error=True)

@serve.deployment(route_prefix="/wholesale")
@serve.ingress(app)
class WholesaleController():

    reward = 0
    finished_observation = False
    timeslot = 0
    
    def __init__(self):
        setup_logging()
        
        self._log = logging.getLogger(__name__)
        self._log.debug("Init wholesale controller")
        # Propably give the values for all timesteps.  TODO: insert dummy variables.
        self.env = Env({})
        #self.env = gym.make(env=env,)


    @app.get("/run")
    def run(self, request: Request):
        self._log.debug(request)
        self.trainer.train()

        

    @app.get("/build_observation")
    def build_observation(self, request: Request):
        obs = request.query_params["obs"]
        self.env._set_obs(obs)
        self._log.info(f"Building observation with: {obs}")
        self.finished_observation = True

        # Testing train loop:
        #self.log_action()
        #self.log_rewards("reward=1")


    @app.get("/check_observation")
    def check_observation(self, request: Request):
        return self.finished_observation


    @app.get("/get_observation")
    def get_observation(self, request: Request):
        try:
            timeslot = request.query_params["timeslot"]
        except:
            pass
        return self.env._get_obs()


    @app.get("/log_action")
    def log_action(self, request: Request):
        #self.trainer.train(self.get_observation())
        self._log.debug("Creating Env6")
        result = self.trainer.train()
        print(pretty_print(result))
        #self.finished_observation = False


    @app.get("/log_rewards")
    def log_rewards(self, request: Request):
        reward = request.query_params["reward"]
        self._log.info(f"Reward logged: {reward}")
        self.env.reward = reward
        self.finished_observation = False
        #return reward

    @app.get("/start_episode")
    def start_episodes(self, request: Request):
        self._log.debug("Creating Env9")
        env_variables = Env_config()
        self._log.debug("Creating Env2")
        config = env_variables.get_rl_config()
        self._log.debug("Creating Env3")
        self._log.info(config)
        
        config["num_gpus"] = 0
        config["num_workers"] = 1
        config["env"] = self.env
        self._log.debug("Creating Env4")
        #self.trainer_cls = get_trainer_class("PPO")
        try:
            self.trainer = PPOTrainer(env=self.env,config=config)
        except Exception as e:
            self._log.error(e)
        
     
    
    @app.get("/end_episode")
    def end_episodes(self, request: Request):
        try:
            register_env("my_env", self.env)
            self.trainer = PPOTrainer(config={"env":"my_env", "observation_space":self.env.observation_space, "action_space": self.env.action_space})
        except Exception as e:
            self._log.error(e)

    @app.get("/reset")
    def reset(self):
        self.env.reset()
        self.finished_observation = False
        pass

    @app.get("/step")
    def step(self, action):    
        self.env.step(action) 
        return 


    





# Uncomment to disable API endpoint
#client = serve.start(detached=True)
WholesaleController.deploy()

#while True:
#    time.sleep(5)