import imp
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
from ray.rllib.agents.impala import impala
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.input_reader import InputReader
import dotenv

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.policy_client import PolicyClient

app = FastAPI()
ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
serve.start()
@serve.deployment(route_prefix="/wholesale")
@serve.ingress(app)
class WholesaleController():

    reward = 0
    finished_observation = False
    timeslot = 0
    episode_ids = []
    i = 0
    
    def __init__(self):
        setup_logging()
        #dotenv.load_dotenv()
        self._log = logging.getLogger(__name__)
        self._log.debug("Init wholesale controller")
        self.action = None
        # Propably give the values for all timesteps.  TODO: insert dummy variables.
        #self.env = Env({})
        #self.env = gym.make(env=env,)


    @app.get("/run")
    def run(self, request: Request):
        
        obs = request.query_params["obs"]
        obs = np.array(json.loads(obs), dtype=np.float32)
        #sample_batch = SampleBatch({"obs": [obs], "rewards": [0], "actions": [0], "dones"=[]})
        #self._log(f"Sample_Batch: {sample_batch}")
        self._log.info(obs)
        if self.action is not None:
            action = self.trainer.compute_single_action(obs)
        else:
            action = self.trainer.compute_single_action(obs,prev_action=self.action, prev_reward=self.reward)
            self.reward = 0
        #training = self.trainer.training_iteration()
        training = self.trainer.train()
        #self.trainer.get_policy().learn_on_batch(sample_batch)
        #self.train_agent()
        self._log.info(action)
        self.action = action
        #return {"action" : action.tolist()}
        #self._log.info(f"Training: {training}")
        #self._log.info(self.trainer.train())
        
        #obs, reward, done, _ = temp_env.step(action[0])
        #self._log.info(f"{obs}, {reward}")
    


    @app.get("/build_observation")
    def build_observation(self, request: Request):
        obs = request.query_params["obs"]
        obs = np.array(json.loads(obs))
        self._log.info("Observation received")
        self.last_obs = obs
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
        
        obs = self.last_obs
        return obs


    @app.get("/log_action")
    def log_action(self, request: Request):
        try:
            episode_id = request.query_params["episode_id"]
        except:
            episode_id = self.episode_ids[-1]
        try:
            obs = request.query_params["obs"]
            obs = np.array(json.loads(obs))
        except:
            obs = self.last_obs
        self.last_action = self.client.log_action(episode_id, obs)
        self._log(self.last_action)
        return self.last_action.tolist()


    @app.get("/get_action")
    def get_action(self, request: Request):
        #self.trainer.train(self.get_observation())
        try:
            episode_id = request.query_params["episode_id"]
        except:
            episode_id = self.episode_ids[-1]
        try:
            obs = request.query_params["obs"]
            obs = np.array(json.loads(obs))
        except:
            obs = self.last_obs
        

        self.last_action = self.client.get_action(episode_id, obs)
        self._log.info(self.last_action)
        return json.dumps(self.last_action.tolist())
        


    @app.get("/log_rewards")
    def log_rewards(self, request: Request):
        reward = float(request.query_params["reward"])
        try:
            episode_id = request.query_params["episode_id"]
        except:
            episode_id = self.episode_ids[-1]
        
        self.client.log_returns(episode_id, reward)
        #episode_id = request.query_params["episode_id"]
        #timeslot = request.query_params["timeslot"] 
        self.finished_observation = False
        self.reward = reward
        self._log.info(f"Reward logged: {reward}")
        


    @app.get("/start_episode")
    def start_episode(self, request: Request):
        
        #self._log.info()
        episode_id = self.client.start_episode()
        self.episode_ids.append(episode_id)
        return episode_id

       
            
        

        
    @app.get("/start_client")
    def start_client(self, request: Request):
        try:
            port = request.query_params["port"]
        except:
            port = 9905

        config = {
            "port" : port,
            "interference_mode" : "remote",
            "no_train" : False

        }
        self.client = PolicyClient(
            f"http://localhost:{config.get('port')}", inference_mode=config.get("interference_mode")
        )
        self._log.info(f"Client startet.")

    
    @app.get("/end_episode")
    def end_episodes(self, request: Request):
        try:
            episode_id = request.query_params["episode_id"]
        except:
            episode_id = self.episode_ids[-1]
        
        self.client.end_episode(episode_id)


    @app.get("/reset")
    def reset(self):
        
        self.finished_observation = False
    





# Uncomment to disable API endpoint
#client = serve.start(detached=True)
WholesaleController.deploy()

#while True:
#    time.sleep(5)