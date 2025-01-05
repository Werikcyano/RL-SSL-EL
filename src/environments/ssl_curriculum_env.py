from rsoccer_gym.ssl.ssl_multi_agent import SSLMultiAgentEnv
import numpy as np
from gym import spaces
from typing import Dict, Tuple

class SSLCurriculumEnv(SSLMultiAgentEnv):
    def __init__(self, curriculum_config=None, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_config = curriculum_config or {}
        self.task_level = curriculum_config.get("initial_task", 0) if curriculum_config else 0
        self.obstacle_pos = np.array([0.0, 0.0])
        
    def reset(self, *, seed=None, options=None) -> Dict:
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
        
        # Recompensa adicional baseada na distância até a bola
        robot_pos = self.get_robot_position(robot_id)
        dist_to_ball = np.linalg.norm(robot_pos - self.ball)
        
        # Penalidade por colisão com obstáculo no nível 2
        obstacle_penalty = 0
        if self.task_level == 2:
            dist_to_obstacle = np.linalg.norm(robot_pos - self.obstacle_pos)
            if dist_to_obstacle < 0.3:  # Raio de colisão
                obstacle_penalty = -1.0
        
        # Recompensa por tempo (quanto mais rápido melhor)
        time_penalty = -0.01
        
        return base_reward + (-dist_to_ball * 0.1) + time_penalty + obstacle_penalty
    
    def get_robot_position(self, robot_id):
        """Retorna a posição do robô como um array numpy"""
        if robot_id.startswith("blue"):
            return self.robot_pos  # Posição definida no reset()
        else:
            # Para robôs amarelos, retorna posição espelhada
            return -self.robot_pos

    def step(self, action_dict):
        observations, rewards, dones, truncated, infos = super().step(action_dict)
        
        # Adiciona a distância até a bola no info de cada agente
        for agent_id in infos.keys():
            if agent_id.startswith("blue"):
                robot_pos = self.get_robot_position(agent_id)
                dist_to_ball = np.linalg.norm(robot_pos - self.ball)
                infos[agent_id]["distance_to_ball"] = dist_to_ball
        
        return observations, rewards, dones, truncated, infos