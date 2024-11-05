import argparse
import os
import yaml
import random
from collections import deque
import numpy as np
from typing import Dict

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)

from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

# RAY_PDB=1 python rllib_multiagent.py
# ray debug
def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        pol_id = "policy_blue"
    elif "yellow" in agent_id:
        pol_id = "policy_yellow"
    return pol_id

@ray.remote
class ScoreCounter:
    def __init__(self, maxlen):
        self.last100 = deque(maxlen=maxlen)
        self.last100.extend([0.0 for _ in range(maxlen)])
        self.maxlen = maxlen

    def append(self, s):
        self.last100.append(s)

    def reset(self):
        self.last100.extend([0.0 for _ in range(self.maxlen)])

    def get_score(self):
        return np.array(self.last100).mean()


@ray.remote
class RewardCounter:
    def __init__(self, maxlen):
        self.r_dist = {
            **{f"blue_{i}": deque(maxlen=maxlen) for i in range(3)},
            **{f"yellow_{i}": deque(maxlen=maxlen) for i in range(3)},
        }
        self.r_speed = {
            **{f"blue_{i}": deque(maxlen=maxlen) for i in range(3)},
            **{f"yellow_{i}": deque(maxlen=maxlen) for i in range(3)},
        }   
        self.r_off = {
            **{f"blue_{i}": deque(maxlen=maxlen) for i in range(3)},
            **{f"yellow_{i}": deque(maxlen=maxlen) for i in range(3)},
        }   
        self.r_def = {
            **{f"blue_{i}": deque(maxlen=maxlen) for i in range(3)},
            **{f"yellow_{i}": deque(maxlen=maxlen) for i in range(3)},
        }   
        self.r_vel_theta = {
            **{f"blue_{i}": deque(maxlen=maxlen) for i in range(3)},
            **{f"yellow_{i}": deque(maxlen=maxlen) for i in range(3)},
        }  
        self.maxlen = maxlen

    def append(self, s: dict):
        for agent in self.r_dist.keys():
            self.r_dist[agent].append(s[agent]["r_dist"])
            self.r_speed[agent].append(s[agent]["r_speed"])
            self.r_off[agent].append(s[agent]["r_off"])
            self.r_def[agent].append(s[agent]["r_def"])
            self.r_vel_theta[agent].append(s[agent]["r_vel_theta"])

    def reset(self):
        agents = list(self.r_dist.keys())
        for agent in agents:
            self.r_dist[agent].extend([0.0 for _ in range(self.maxlen)])
            self.r_speed[agent].extend([0.0 for _ in range(self.maxlen)])
            self.r_off[agent].extend([0.0 for _ in range(self.maxlen)])
            self.r_def[agent].extend([0.0 for _ in range(self.maxlen)])
            self.r_vel_theta[agent].extend([0.0 for _ in range(self.maxlen)])

    def get_reward(self):
        rewards = {
            agent: {
                "r_dist": np.array(self.r_dist[agent]).mean(),
                "r_speed": np.array(self.r_speed[agent]).mean(),
                "r_off": np.array(self.r_off[agent]).mean(),
                "r_def": np.array(self.r_def[agent]).mean(),
                "r_vel_theta": np.array(self.r_vel_theta[agent]).mean(),
            } for agent in self.r_dist.keys()
        }
        return rewards
    

class SelfPlayUpdateCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):

        super().__init__(legacy_callbacks_dict)

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):

        episode.hist_data["score"] = []

    def on_episode_end(
        self, *, worker, base_env, policies, episode: Episode, **kwargs
    ) -> None:
        info_a = episode.last_info_for("blue_0")
        single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]

        score_counter = ray.get_actor("score_counter")
        score_counter.append.remote(single_score)

        info_geral = {}
        for agent in ["blue_0", "blue_1", "blue_2", "yellow_0", "yellow_1", "yellow_2"]:
            info_geral[agent] = episode.last_info_for(agent)
        reward_counter = ray.get_actor("reward_counter")
        reward_counter.append.remote(info_geral)

    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when score is high enough
        """
        score_counter = ray.get_actor("score_counter")
        current_score = ray.get(score_counter.get_score.remote())

        info["result"]["custom_metrics"]["score"] = current_score


        reward_counter = ray.get_actor("reward_counter")
        current_reward = ray.get(reward_counter.get_reward.remote())
        for agent in ["blue_0", "blue_1", "blue_2", "yellow_0", "yellow_1", "yellow_2"]:
            info["result"]["custom_metrics"][agent] = {}
            info["result"]["custom_metrics"][agent]["r_dist"] = current_reward[agent]["r_dist"]
            info["result"]["custom_metrics"][agent]["r_speed"] = current_reward[agent]["r_speed"]
            info["result"]["custom_metrics"][agent]["r_off"] = current_reward[agent]["r_off"]
            info["result"]["custom_metrics"][agent]["r_def"] = current_reward[agent]["r_def"]
            info["result"]["custom_metrics"][agent]["r_vel_theta"] = current_reward[agent]["r_vel_theta"]

        if current_score > 0.6:
            print("---- Updating Opponent!!! ----")
            algorithm = info["algorithm"]
            algorithm.set_weights(
                {
                    "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                }
            )
            score_counter = ray.get_actor("score_counter")
            score_counter.restart.remote()


if __name__ == "__main__":
    ray.init()

    with open("config.yaml") as f:
        # use safe_load instead load
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}

    counter = ScoreCounter.options(name="score_counter").remote(
        maxlen=file_configs["score_average_over"]
    )

    reward_counter = RewardCounter.options(name="reward_counter").remote(
        maxlen=file_configs["score_average_over"]
    )

    configs["env_config"] = file_configs["env"]

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    # Register the models to use.
    ModelCatalog.register_custom_action_dist("beta_dist", TorchBetaTest)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
    # Each policy can have a different configuration (including custom model).


    configs["callbacks"] = SelfPlayUpdateCallback
    configs["multiagent"] = {
        "policies": {
            "policy_blue": (None, obs_space, act_space, {}),
            "policy_yellow": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["policy_blue"],
    }
    configs["model"] = {
        "custom_model": "custom_vf_model",
        "custom_model_config": file_configs["custom_model"],
        "custom_action_dist": "beta_dist",
    }
    configs["env"] = "Soccer"

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config=configs,
        stop={
            "timesteps_total": int(file_configs["timesteps_total"]),
            # "time_total_s": 7200, #2h
            # "time_total_s": 60*60, #1h
        },
        checkpoint_freq=int(file_configs["checkpoint_freq"]),
        checkpoint_at_end=True,
        local_dir=os.path.abspath("volume"),
        #resume=True,
        restore=file_configs["checkpoint_restore"],
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

