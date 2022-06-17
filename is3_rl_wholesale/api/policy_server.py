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

from is3_rl_wholesale.api.wholesale_env import Env_config
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.agents.impala.impala import ImpalaTrainer
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray import tune
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator
from ray.rllib.agents.trainer import with_common_config


def start_server(server_address = "localhost",port = 9905):
    _log = logging.getLogger(__name__)
    SERVER_ADDRESS = server_address
    SERVER_BASE_PORT = port  # + worker-idx - 1

    _log.info("Starting Server..")
    #ray.init()
    #serve.start(detached=True)
    def _input(ioctx):
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                SERVER_BASE_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None

    observation_space, action_space = Env_config().get_gym_spaces()

    config = {
        "env": None,
        "observation_space" : observation_space,
        "action_space" : action_space,
        "input" : _input,
        "log_level": "DEBUG",
        "model": {},
        "train_batch_size": 6,
        "input_evaluation" : [],
        "sgd_minibatch_size" : 2,
        "off_policy_estimation_methods": {},
    }
        
    config.update(
                {
                    "num_gpus": 0,
                    "model": {"use_lstm": True},
                }
            )

    config = {
        # Indicate that the Algorithm we setup here doesn't need an actual env.
        # Allow spaces to be determined by user (see below).
        "env": None,
        # TODO: (sven) make these settings unnecessary and get the information
        #  about the env spaces from the client.
        "observation_space" : observation_space,
        "action_space" : action_space,
        # Use the `PolicyServerInput` to generate experiences.
        "input": _input,
        # Use n worker processes to listen on different ports.
        "num_workers": 2,
        # Disable OPE, since the rollouts are coming from online clients.
        #"off_policy_estimation_methods": {},
        # Create a "chatty" client/server or not.
        #"callbacks": MyCallbacks if args.callbacks_verbose else None,
        # DL framework to use.
        # Set to INFO so we'll see the server's actual address:port.
        "log_level": "INFO",
        "model": {"fcnet_hiddens": [512, 512]},
        "rollout_fragment_length": 1,
        #"sgd_minibatch_size" : 1,
        "train_batch_size": 1,
        "input_evaluation": [],
        "input_evaluation" : [],
        "num_gpus": 0,
        "model": {"use_lstm": False},
    }
    DEFAULT_CONFIG = with_common_config({
    # V-trace params (see vtrace_tf/torch.py).
    "vtrace": True,
    "observation_space" : observation_space,
    "action_space" : action_space,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    # If True, drop the last timestep for the vtrace calculations, such that
    # all data goes into the calculations as [B x T-1] (+ the bootstrap value).
    # This is the default and legacy RLlib behavior, however, could potentially
    # have a destabilizing effect on learning, especially in sparse reward
    # or reward-at-goal environments.
    # False for not dropping the last timestep.
    "vtrace_drop_last_ts": True,
    "input": _input,
    "input_evaluation" : [],
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
    "train_batch_size": 1,
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
    "num_multi_gpu_tower_stacks": 1,
    # How many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 1,
    # Number of passes to make over each train batch.
    "num_sgd_iter": 1,
    # Set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.2,
    # Number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    "replay_buffer_num_slots": 10,
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

    # DEPRECATED:
    #"num_data_loader_buffers": DEPRECATED_VALUE,
    })
    #trainer_cls = get_trainer_class("A3C")
    #trainer = trainer_cls(config=config)
    #trainer = PPOTrainer(config=config)
    ts = 0
    _log.info("Server started.")
    #while True:
    #DEFAULT_CONFIG["gpus"] = 0
        
    #    trainer.train()
    tune.run("IMPALA", config=DEFAULT_CONFIG, verbose=2)#, restore=checkpoint_path)
    
    #while True:
    #    time.sleep(5)
    #policy_server = policy_server_input(,port=8000)
    #policy_server