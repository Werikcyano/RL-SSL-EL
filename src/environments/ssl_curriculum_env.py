from rsoccer_gym.ssl.ssl_multi_agent import SSLMultiAgentEnv
import numpy as np
from gym import spaces
from typing import Dict, Tuple
from time import time

class SSLCurriculumEnv(SSLMultiAgentEnv):
    def __init__(self, curriculum_config=None, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_config = curriculum_config or {}
        self.task_level = curriculum_config.get("initial_task", 0) if curriculum_config else 0
        self.obstacle_pos = np.array([0.0, 0.0])
        self.ball_touched = False  # Flag para controlar se a bola foi tocada
        
        # Métricas de continuidade
        self.continuity_metrics = {
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0
        }
        
    def reset(self, *, seed=None, options=None) -> Dict:
        # Reseta flag de toque na bola
        self.ball_touched = False
        
        # Reseta métricas de continuidade
        self.continuity_metrics.update({
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0
        })
        
        if seed is not None:
            np.random.seed(seed)
            
        if self.task_level == 0:
            # Tarefa 1: Posições fixas
            self.ball = np.array([1.0, 1.0])
            self.robot_pos = np.array([-1.0, -1.0])
            
        elif self.task_level == 1:
            # Tarefa 2: Posições aleatórias
            self.ball = np.random.uniform(-2, 2, 2)
            self.robot_pos = np.random.uniform(-2, 2, 2)
            
        elif self.task_level == 2:
            # Tarefa 3: Posições aleatórias + obstáculo
            self.ball = np.random.uniform(-2, 2, 2)
            self.robot_pos = np.random.uniform(-2, 2, 2)
            
            # Posiciona obstáculo entre robô e bola
            direction = self.ball - self.robot_pos
            self.obstacle_pos = self.robot_pos + direction * 0.5
                
        return super().reset(seed=seed, options=options)
        
    def compute_reward(self, robot_id: str) -> float:
        base_reward = super().compute_reward(robot_id)
        
        if not robot_id.startswith("blue"):
            return 0  # No nível 0, apenas o time azul recebe recompensas
        
        # Recompensa adicional baseada na distância até a bola
        robot_pos = self.get_robot_position(robot_id)
        dist_to_ball = np.linalg.norm(robot_pos - self.ball)
        
        # Verifica se tocou na bola (nível 0)
        reward = 0
        if self.task_level == 0:
            task_config = self.curriculum_config["tasks"].get(str(self.task_level)) or self.curriculum_config["tasks"].get(self.task_level)
            if task_config and dist_to_ball <= task_config.get("success_distance", 0.2):
                if not self.ball_touched:  # Apenas recompensa na primeira vez que tocar
                    reward = task_config.get("reward_touch", 10.0)
                    self.ball_touched = True
        
        # Penalidade por colisão com obstáculo no nível 2
        obstacle_penalty = 0
        if self.task_level == 2:
            dist_to_obstacle = np.linalg.norm(robot_pos - self.obstacle_pos)
            if dist_to_obstacle < 0.3:  # Raio de colisão
                obstacle_penalty = -1.0
        
        # Recompensa por tempo (quanto mais rápido melhor)
        time_penalty = -0.01
        
        return base_reward + (-dist_to_ball * 0.1) + time_penalty + obstacle_penalty + reward
    
    def get_robot_position(self, robot_id):
        """Retorna a posição do robô como um array numpy"""
        if robot_id.startswith("blue"):
            return self.robot_pos  # Posição definida no reset()
        else:
            # Para robôs amarelos, retorna posição espelhada
            return -self.robot_pos

    def track_reset(self, reset_type: str):
        """
        Atualiza as métricas quando ocorre um reset
        Args:
            reset_type: 'lateral' ou 'endline'
        """
        current_time = self.steps / self.fps
        
        # Atualiza contadores de reset
        self.continuity_metrics['total_resets'] += 1
        self.continuity_metrics[f'{reset_type}_resets'] += 1
        
        # Calcula e registra tempo entre resets
        if self.continuity_metrics['last_reset_time'] is not None:
            time_between = current_time - self.continuity_metrics['last_reset_time']
            self.continuity_metrics['time_between_resets'].append(time_between)
            
            # Atualiza sequência máxima sem reset
            if time_between > self.continuity_metrics['max_sequence_without_reset']:
                self.continuity_metrics['max_sequence_without_reset'] = time_between
        
        self.continuity_metrics['last_reset_time'] = current_time
        self.continuity_metrics['current_sequence_time'] = 0

    def step(self, action_dict):
        observations, rewards, dones, truncated, infos = super().step(action_dict)
        
        # Atualiza tempo da sequência atual
        current_time = self.steps / self.fps
        if self.continuity_metrics['last_reset_time'] is None:
            self.continuity_metrics['current_sequence_time'] = current_time
        else:
            self.continuity_metrics['current_sequence_time'] = current_time - self.continuity_metrics['last_reset_time']
        
        # Adiciona métricas de continuidade ao info de cada agente
        for agent_id in infos.keys():
            if agent_id.startswith("blue"):
                # Calcula a distância até a bola para cada robô azul
                robot_pos = np.array([self.frame.robots_blue[int(agent_id.split('_')[1])].x,
                                    self.frame.robots_blue[int(agent_id.split('_')[1])].y])
                ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
                dist_to_ball = np.linalg.norm(robot_pos - ball_pos)
                
                infos[agent_id].update({
                    "distance_to_ball": dist_to_ball,
                    "ball_touched": self.ball_touched,
                    "continuity_metrics": self.continuity_metrics.copy()
                })
                
                # No nível 0, termina o episódio quando tocar na bola
                if self.task_level == 0 and self.ball_touched:
                    dones[agent_id] = True
                    dones["__all__"] = True
        
        return observations, rewards, dones, truncated, infos