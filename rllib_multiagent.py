import argparse
import os
import yaml
from collections import deque
import numpy as np
from typing import Dict

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation.episode import Episode

from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv, SSLMultiAgentEnv_record
from src.environments.ssl_curriculum_env import SSLCurriculumEnv
from src.callbacks.curriculum_callback import CurriculumCallback

from torch.utils.tensorboard import SummaryWriter
import os

# RAY_PDB=1 python rllib_multiagent.py
# ray debug

def create_curriculum_env(config):
    # Mantém a configuração original do curriculum se estiver presente
    if "curriculum" in config and config["curriculum"].get("enabled", False):
        task_level = config["curriculum"].get("initial_task", 0)
        task_config = config["curriculum"]["tasks"].get(str(task_level)) or config["curriculum"]["tasks"].get(task_level)
        if task_config:
            num_blue = task_config.get("num_agents_blue", 3)
            num_yellow = task_config.get("num_agents_yellow", 0)
            
            # Ajusta init_pos baseado no número de agentes
            if "init_pos" in config:
                blue_pos = {}
                yellow_pos = {}
                
                # Garante que temos pelo menos uma posição para cada time
                for i in range(num_blue):
                    key = str(i + 1)
                    if key in config["init_pos"]["blue"]:
                        blue_pos[key] = config["init_pos"]["blue"][key]
                    else:
                        blue_pos[key] = [-0.5 - i, 0.0, 0.0]  # Posição padrão
                
                for i in range(num_yellow):
                    key = str(i + 1)
                    if key in config["init_pos"]["yellow"]:
                        yellow_pos[key] = config["init_pos"]["yellow"][key]
                    else:
                        yellow_pos[key] = [1.5 + i, 0.0, 180.0]  # Posição padrão
                
                config["init_pos"] = {
                    "blue": blue_pos,
                    "yellow": yellow_pos if num_yellow > 0 else {},
                    "ball": config["init_pos"]["ball"]
                }
    
    return SSLCurriculumEnv(**config)

def create_rllib_env_recorder(config):
    trigger = lambda t: t % 1 == 0
    config["render_mode"] = "rgb_array"
    config = config.copy()
    
    # Se temos configuração de curriculum, mostra a task atual
    if config.get("curriculum", {}).get("enabled", False):
        task_level = config["curriculum"].get("initial_task", 0)
        print(f"\n[AVALIAÇÃO] Task Atual: {task_level}")
    
    ssl_el_env = SSLCurriculumEnv(**config)
    return SSLMultiAgentEnv_record(ssl_el_env, video_folder="/ws/videos", episode_trigger=trigger, disable_logger=True)

def create_rllib_env(config):
    return SSLCurriculumEnv(**config)

def create_policy_mapping_fn(curriculum_enabled=False, curriculum_config=None):
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if "blue" in agent_id:
            return "policy_blue"
        elif "yellow" in agent_id:
            # Se estamos usando curriculum e estamos na Tarefa 1, retorna None para não gerar ações
            if curriculum_enabled and curriculum_config:
                task_level = curriculum_config.get("initial_task", 0)
                task_config = curriculum_config["tasks"].get(str(task_level)) or curriculum_config["tasks"].get(task_level)
                if task_level == 1:
                    return None  # Robôs amarelos não receberão ações na Tarefa 1
            return "policy_yellow"
        return "policy_blue"  # Caso padrão
    return policy_mapping_fn

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
    

class SelfPlayUpdateCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.continuity_stats = {
            'episode_resets': [],
            'avg_time_between_resets': [],
            'max_sequence_lengths': [],
            'reset_distribution': {
                'lateral': [],
                'endline': []
            }
        }

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):
        episode.hist_data["score"] = []

    def on_episode_end(
        self, *, worker, base_env, policies, episode: Episode, **kwargs
    ) -> None:
        # Processa métricas de continuidade
        metrics = episode.last_info_for("blue_0").get("continuity_metrics", {})
        if metrics:
            self.continuity_stats['episode_resets'].append(metrics['total_resets'])
            
            # Calcula média de tempo entre resets
            if metrics['time_between_resets']:
                avg_time = np.mean(metrics['time_between_resets'])
                self.continuity_stats['avg_time_between_resets'].append(avg_time)
            
            # Registra sequência máxima sem reset
            self.continuity_stats['max_sequence_lengths'].append(
                metrics['max_sequence_without_reset']
            )
            
            # Registra distribuição dos tipos de reset
            self.continuity_stats['reset_distribution']['lateral'].append(
                metrics['lateral_resets']
            )
            self.continuity_stats['reset_distribution']['endline'].append(
                metrics['endline_resets']
            )
            
            # Adiciona métricas customizadas ao episódio
            episode.custom_metrics.update({
                "total_resets": metrics['total_resets'],
                "avg_time_between_resets": avg_time if metrics['time_between_resets'] else 0,
                "max_sequence_without_reset": metrics['max_sequence_without_reset'],
                "lateral_resets_ratio": metrics['lateral_resets'] / max(metrics['total_resets'], 1),
                "endline_resets_ratio": metrics['endline_resets'] / max(metrics['total_resets'], 1),
                "goals_blue": metrics['goals_blue'],
                "goals_yellow": metrics['goals_yellow'],
                "total_goals": metrics['total_goals'],
                "goals_per_episode": metrics['goals_per_episode']
            })

        # Processa métricas de score
        info_a = episode.last_info_for("blue_0")
        single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]

        score_counter = ray.get_actor("score_counter")
        score_counter.append.remote(single_score)

    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when score is high enough
        """
        score_counter = ray.get_actor("score_counter")
        current_score = ray.get(score_counter.get_score.remote())

        info["result"]["custom_metrics"]["score"] = current_score

        if current_score > 0.6:
            print("---- Updating Opponent!!! ----")
            algorithm = info["algorithm"]
            algorithm.set_weights(
                {
                    "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                }
            )
            score_counter = ray.get_actor("score_counter")
            score_counter.reset.remote()

parser = argparse.ArgumentParser(description="Treina multiagent SSL-EL.")
parser.add_argument("--evaluation", action="store_true", help="Irá renderizar um episódio de tempos em tempos.")
parser.add_argument("--curriculum", action="store_true", help="Ativa o modo curriculum learning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina multiagent SSL-EL.")
    parser.add_argument("--evaluation", action="store_true", help="Irá renderizar um episódio de tempos em tempos.")
    parser.add_argument("--curriculum", action="store_true", help="Ativa o modo curriculum learning")
    args = parser.parse_args()

    ray.init()

    with open("config.yaml") as f:
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    configs["env"] = "Soccer"
    
    # Configura o ambiente baseado no modo de treinamento
    use_curriculum = args.curriculum and file_configs["curriculum"]["enabled"]
    
    if use_curriculum:
        print("\nIniciando treinamento com Curriculum Learning...")
        configs["env_config"] = {
            **file_configs["env"],
            "curriculum": file_configs["curriculum"]
        }
        configs["callbacks"] = CurriculumCallback
    else:
        print("\nIniciando treinamento com Self-Play padrão...")
        configs["env_config"] = file_configs["env"]
        configs["callbacks"] = SelfPlayUpdateCallback
    
    # Registra o ambiente apropriado
    tune.registry.register_env("Soccer", create_curriculum_env)
    temp_env = create_curriculum_env(configs["env_config"].copy())

    # Configuração de avaliação
    if args.evaluation:
        eval_configs = file_configs["evaluation"].copy()
        env_config_eval = file_configs["env"].copy()
        
        # Se estiver usando curriculum, propaga a configuração para avaliação
        if use_curriculum:
            env_config_eval["curriculum"] = file_configs["curriculum"]
            
            # Ajusta as posições iniciais baseado no número de agentes no nível atual
            task_level = file_configs["curriculum"]["initial_task"]
            task_config = file_configs["curriculum"]["tasks"].get(str(task_level)) or file_configs["curriculum"]["tasks"].get(task_level)
            
            num_blue = task_config.get("num_agents_blue", 3)
            num_yellow = task_config.get("num_agents_yellow", 0)
            
            blue_pos = {}
            yellow_pos = {}
            
            for i in range(num_blue):
                key = str(i + 1)
                if key in env_config_eval["init_pos"]["blue"]:
                    blue_pos[key] = env_config_eval["init_pos"]["blue"][key]
            
            for i in range(num_yellow):
                key = str(i + 1)
                if key in env_config_eval["init_pos"]["yellow"]:
                    yellow_pos[key] = env_config_eval["init_pos"]["yellow"][key]
            
            env_config_eval["init_pos"] = {
                "blue": blue_pos,
                "yellow": yellow_pos,
                "ball": env_config_eval["init_pos"]["ball"]
            }
        
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] = eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = eval_configs["evaluation_config"].copy()
        configs["evaluation_config"]["env_config"] = env_config_eval
        tune.registry.register_env("Soccer_recorder", create_rllib_env_recorder)

    # Configuração dos espaços de observação e ação
    if use_curriculum:
        task_level = file_configs["curriculum"]["initial_task"]
        task_config = file_configs["curriculum"]["tasks"].get(str(task_level)) or file_configs["curriculum"]["tasks"].get(task_level)
        num_blue = task_config.get("num_agents_blue", 3)
        if num_blue > 0:
            obs_space = temp_env.observation_space["blue_0"]
            act_space = temp_env.action_space["blue_0"]
        else:
            raise ValueError("Número de agentes azuis deve ser maior que 0")
    else:
        obs_space = temp_env.observation_space["blue_0"]
        act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    # Registra os modelos customizados
    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

    # Configuração multiagente
    if use_curriculum:
        task_level = file_configs["curriculum"]["initial_task"]
        task_config = file_configs["curriculum"]["tasks"].get(str(task_level)) or file_configs["curriculum"]["tasks"].get(task_level)
        
        # Configuração básica das políticas
        policies = {
            "policy_blue": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}}),
        }
        
        # Adiciona política amarela apenas se não estiver na Tarefa 1
        if task_level != 1:
            policies["policy_yellow"] = (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_yellow'}})
        
        configs["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": create_policy_mapping_fn(True, file_configs["curriculum"]),
            "policies_to_train": ["policy_blue"],
        }
    else:
        # Configuração padrão para selfplay
        configs["multiagent"] = {
            "policies": {
                "policy_blue": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}}),
                "policy_yellow": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_yellow'}}),
            },
            "policy_mapping_fn": create_policy_mapping_fn(),
            "policies_to_train": ["policy_blue"],
        }

    # Configuração do modelo
    configs["model"] = {
        "custom_model": "custom_vf_model",
        "custom_model_config": file_configs["custom_model"],
        "custom_action_dist": "beta_dist",
    }

    # Contador de pontuação
    counter = ScoreCounter.options(name="score_counter").remote(
        maxlen=file_configs["score_average_over"]
    )

    # Executa o treinamento
    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config=configs,
        stop={
            "timesteps_total": int(file_configs["timesteps_total"]),
        },
        checkpoint_freq=int(file_configs["checkpoint_freq"]),
        checkpoint_at_end=True,
        local_dir="/ws/volume",
        restore=file_configs["checkpoint_restore"],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
