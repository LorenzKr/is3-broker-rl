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


app = FastAPI()
ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
serve.start()
@serve.deployment(route_prefix="/wholesale")
@serve.ingress(app)
class WholesaleController():

    reward = 0
    finished_observation = False
    timeslot = 0
    i = 0
    
    def __init__(self):
        setup_logging()
        
        self._log = logging.getLogger(__name__)
        self._log.debug("Init wholesale controller")
        self.action = None
        # Propably give the values for all timesteps.  TODO: insert dummy variables.
        #self.env = Env({})
        #self.env = gym.make(env=env,)


    @app.get("/run")
    def run(self, request: Request):
        
        #self._log.info(self.trainer.train())
        #return
        temp_env = Env({})
        self._log.info(np.shape(temp_env._get_obs()))
        config = Env_config().observation_space
        self._log.info(config.contains(temp_env._get_obs()))
        #self.trainer.get_policy().
        try:
            if self.action is not None:
                action = self.trainer.compute_single_action(temp_env._get_obs())
            else:
                action = self.trainer.compute_single_action(temp_env._get_obs(),prev_action=self.action, prev_reward=1)
            #training = self.trainer.training_iteration()
            training = self.trainer.train()
            self.action = action
            self._log.info(f"Training: {training}")
            #self._log.info(self.trainer.train())
            
            #obs, reward, done, _ = temp_env.step(action[0])
            #self._log.info(f"{obs}, {reward}")
        except Exception as e:
            self._log.error(e)

    @app.get("/test")
    def test(self, request: Request):
        test3 = self.trainer.train()
        training = self.trainer._episode_history
        self._log.info(f"Training: {training}")
        self._log.info(f"Training2: {test3}")

        

    @app.get("/build_observation")
    def build_observation(self, request: Request):
        obs = request.query_params["obs"]
        obs = np.array(json.loads(obs))
        self._log.info(type(obs))
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
        #result = self.trainer.train()
        #print(pretty_print(result))
        try:
            self.env.log_action(self.env._get_obs())
        except Exception as e:
            self._log.error(e)
        #self.finished_observation = False

    @app.get("/get_action")
    def get_action(self, request: Request):
        #self.trainer.train(self.get_observation())
        episode_id = request.query_params["episode_id"]
        # Temp
        obs = request.query_params["obs"]
        self._log.debug("Creating Env6")
        #self.trainer.compute_action()
        #result = self.trainer.train()
        #print(pretty_print(result))
        try:
            action = self.trainer.train()#self.env.get_action(obs, episode_id)
            self._log.info(f"Taking Action {action}")
            return action
        except Exception as e:
            self._log.error(e)


    @app.get("/log_rewards")
    def log_rewards(self, request: Request):
        reward = request.query_params["reward"]
        episode_id = request.query_params["episode_id"]
        #self.env.reward = reward
        self.env.log_returns(reward)
        self.finished_observation = False
        self._log.info(f"Reward logged: {reward}")
        #return reward

    @app.get("/start_episode")
    def start_episode(self, request: Request):
        self._log.debug("Creating Env9")
        env_variables = Env_config()
        self._log.debug("Creating Env2")
        config = env_variables.get_rl_config()
        self._log.debug("Creating Env3")
        self._log.info(config)
        
        
        self._log.debug("Creating Env4")
        def env_creator(env_config):
            env = Env({})
            return env
        
        register_env("my_env", env_creator)
        #config["env"] = "my_env"
        random_int = int(np.random.random_integers(1000))
        #self.env= env_creator({})
        #self._log.info(f"{self.env}")
        #self.trainer_cls = get_trainer_class("PPO")
        DEFAULT_CONFIG = ({
            # Should use a critic as a baseline (otherwise don't use value baseline;
            # required for using GAE).
            "use_critic": True,
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # Size of rollout batch
            "rollout_fragment_length": 10,
            # GAE(gamma) parameter
            "lambda": 1.0,
            # Max global norm for each gradient calculated by worker
            "grad_clip": 40.0,
            # Learning rate
            "lr": 0.0001,
            # Learning rate schedule
            "lr_schedule": None,
            # Value Function Loss coefficient
            "vf_loss_coeff": 0.5,
            # Entropy coefficient
            "entropy_coeff": 0.01,
            # Entropy coefficient schedule
            "entropy_coeff_schedule": None,
            # Min time (in seconds) per reporting.
            # This causes not every call to `training_iteration` to be reported,
            # but to wait until n seconds have passed and then to summarize the
            # thus far collected results.
            "min_time_s_per_reporting": 5,
            # Workers sample async. Note that this increases the effective
            # rollout_fragment_length by up to 5x due to async buffering of batches.
            "sample_async": True,

            # Use the Trainer's `training_iteration` function instead of `execution_plan`.
            # Fixes a severe performance problem with A3C. Setting this to True leads to a
            # speedup of up to 3x for a large number of workers and heavier
            # gradient computations (e.g. ray/rllib/tuned_examples/a3c/pong-a3c.yaml)).
            "_disable_execution_plan_api": True,
        })
       
        DEFAULT_CONFIG["env"] = "my_env"
        DEFAULT_CONFIG["train_batch_size"] = 1
        try:
            self.trainer = A3CTrainer(config=DEFAULT_CONFIG)
            self._log.info(f"{self.trainer.env_creator}")
            #self.episode_id = self.env.start_episode()#f"testId{random_int}")
            #self.trainer = PPOTrainer(env=self.env,config=config)
            return #self.episode_id
        except Exception as e:
            self._log.error(e)
            #self._log.info(f"Episode_id {self.episode_id}")
        
        
    
    
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


"""
class _input(InputReader):
    def __init__(self) -> None:
        super().__init__()



    def input(IOContext):


        return InputReader().

"""




# Uncomment to disable API endpoint
#client = serve.start(detached=True)
WholesaleController.deploy()

#while True:
#    time.sleep(5)