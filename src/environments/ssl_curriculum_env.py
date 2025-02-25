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
        self.last_ball_pos = None  # Para detectar movimento da bola
        self.ball_possession_blue = 0  # Contador de posse de bola para time azul
        self.ball_possession_yellow = 0  # Contador de posse de bola para time amarelo
        
        # Métricas de continuidade
        self.continuity_metrics = {
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0,
            'ball_possession_blue': 0,
            'ball_possession_yellow': 0
        }
        
    def reset(self, *, seed=None, options=None) -> Dict:
        # Reseta flags e contadores
        self.ball_touched = False
        self.last_ball_pos = None
        self.ball_possession_blue = 0
        self.ball_possession_yellow = 0
        
        # Reseta métricas de continuidade
        self.continuity_metrics.update({
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0,
            'ball_possession_blue': 0,
            'ball_possession_yellow': 0
        })
        
        if seed is not None:
            np.random.seed(seed)
            
        task_config = self.curriculum_config["tasks"].get(str(self.task_level)) or self.curriculum_config["tasks"].get(self.task_level)
            
        if self.task_level == 0:
            # Tarefa 0: Posições fixas do config
            return super().reset(seed=seed, options=options)
            
        elif self.task_level == 1:
            # Tarefa 1: Posições semi-aleatórias mantendo formação tática
            if task_config and "init_pos" in task_config:
                # Usa posições base do config com pequena variação aleatória
                for team in ["blue", "yellow"]:
                    if team in task_config["init_pos"]:
                        for robot_id, pos in task_config["init_pos"][team].items():
                            # Adiciona pequena variação aleatória nas posições (±0.3 metros)
                            pos[0] += np.random.uniform(-0.3, 0.3)
                            pos[1] += np.random.uniform(-0.3, 0.3)
                            # Mantém o ângulo original
                
                # Posição da bola com variação controlada
                ball_x = np.random.uniform(-1.0, 1.0)
                ball_y = np.random.uniform(-1.0, 1.0)
                task_config["init_pos"]["ball"] = [ball_x, ball_y]
                
            return super().reset(seed=seed, options=options)
            
        elif self.task_level == 2:
            # Tarefa 2: Mantém implementação existente
            self.ball = np.random.uniform(-2, 2, 2)
            self.robot_pos = np.random.uniform(-2, 2, 2)
            direction = self.ball - self.robot_pos
            self.obstacle_pos = self.robot_pos + direction * 0.5
            return super().reset(seed=seed, options=options)
                
        return super().reset(seed=seed, options=options)
        
    def compute_reward(self, robot_id: str) -> float:
        base_reward = super().compute_reward(robot_id)
        
        if not robot_id.startswith("blue"):
            return 0
        
        # Obtém posições atuais
        robot_pos = self.get_robot_position(robot_id)
        ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
        dist_to_ball = np.linalg.norm(robot_pos - ball_pos)
        
        reward = 0
        task_config = self.curriculum_config["tasks"].get(str(self.task_level)) or self.curriculum_config["tasks"].get(self.task_level)
        
        if self.task_level == 0:
            # Lógica existente para Tarefa 0
            if task_config and dist_to_ball <= task_config.get("success_distance", 0.2):
                if not self.ball_touched:
                    reward = task_config.get("reward_touch", 10.0)
                    self.ball_touched = True
                    
        elif self.task_level == 1:
            # Nova lógica para Tarefa 1
            # 1. Recompensa por proximidade da bola
            ball_reward = -0.1 * dist_to_ball
            
            # 2. Recompensa por posse de bola
            possession_reward = 0
            if dist_to_ball < 0.2:  # Distância de posse
                self.ball_possession_blue += 1
                possession_reward = 0.1 * min(self.ball_possession_blue / 30, 1.0)  # Máximo após 1 segundo (30 frames)
            
            # 3. Penalidade se oponente está próximo da bola
            opponent_penalty = 0
            if self.n_robots_yellow > 0:
                yellow_pos = np.array([self.frame.robots_yellow[0].x, self.frame.robots_yellow[0].y])
                yellow_dist_to_ball = np.linalg.norm(yellow_pos - ball_pos)
                if yellow_dist_to_ball < dist_to_ball:  # Oponente mais próximo da bola
                    opponent_penalty = -0.2
                
            # 4. Recompensa por manter bola longe do oponente
            positioning_reward = 0
            if self.n_robots_yellow > 0:
                yellow_pos = np.array([self.frame.robots_yellow[0].x, self.frame.robots_yellow[0].y])
                ball_to_yellow_dist = np.linalg.norm(yellow_pos - ball_pos)
                positioning_reward = 0.05 * min(ball_to_yellow_dist, 2.0)  # Limitado a 2 metros
            
            # Combina todas as recompensas
            reward = ball_reward + possession_reward + opponent_penalty + positioning_reward
            
        elif self.task_level == 2:
            # Mantém lógica existente para Tarefa 2
            dist_to_obstacle = np.linalg.norm(robot_pos - self.obstacle_pos)
            if dist_to_obstacle < 0.3:
                reward -= 1.0
        
        # Penalidade por tempo comum a todas as tarefas
        time_penalty = -0.01
        
        return base_reward + reward + time_penalty
    
    def get_robot_position(self, robot_id):
        """Retorna a posição do robô como um array numpy"""
        if robot_id.startswith("blue"):
            robot_num = int(robot_id.split("_")[1])
            return np.array([self.frame.robots_blue[robot_num].x, 
                           self.frame.robots_blue[robot_num].y])
        else:
            robot_num = int(robot_id.split("_")[1])
            return np.array([self.frame.robots_yellow[robot_num].x, 
                           self.frame.robots_yellow[robot_num].y])

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
        
        # Atualiza métricas de posse de bola
        self.continuity_metrics['ball_possession_blue'] = self.ball_possession_blue
        self.continuity_metrics['ball_possession_yellow'] = self.ball_possession_yellow

    def step(self, action_dict):
        observations, rewards, dones, truncated, infos = super().step(action_dict)
        
        # Atualiza tempo da sequência atual
        current_time = self.steps / self.fps
        if self.continuity_metrics['last_reset_time'] is None:
            self.continuity_metrics['current_sequence_time'] = current_time
        else:
            self.continuity_metrics['current_sequence_time'] = current_time - self.continuity_metrics['last_reset_time']
        
        # Verifica toque na bola (Task 0)
        if self.task_level == 0:
            current_ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
            
            # Verifica se algum robô azul está próximo da bola
            for i in range(self.n_robots_blue):
                robot_pos = np.array([self.frame.robots_blue[i].x, self.frame.robots_blue[i].y])
                dist_to_ball = np.linalg.norm(robot_pos - current_ball_pos)
                
                task_config = self.curriculum_config["tasks"].get(str(self.task_level)) or self.curriculum_config["tasks"].get(self.task_level)
                if task_config and dist_to_ball <= task_config.get("success_distance", 0.2):
                    # Verifica se a bola se moveu
                    if self.last_ball_pos is not None:
                        ball_movement = np.linalg.norm(current_ball_pos - self.last_ball_pos)
                        if ball_movement > 0.01:  # Threshold para movimento da bola
                            self.ball_touched = True
            
            self.last_ball_pos = current_ball_pos
        
        # Adiciona métricas de continuidade ao info de cada agente
        for agent_id in infos.keys():
            if agent_id.startswith("blue"):
                # Calcula a distância até a bola para cada robô azul
                robot_pos = np.array([self.frame.robots_blue[int(agent_id.split('_')[1])].x,
                                    self.frame.robots_blue[int(agent_id.split('_')[1])].y])
                ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
                dist_to_ball = np.linalg.norm(robot_pos - ball_pos)
                
                # Atualiza métricas específicas da Tarefa 1
                if self.task_level == 1:
                    if dist_to_ball < 0.2:  # Distância de posse
                        self.ball_possession_blue += 1
                    infos[agent_id].update({
                        "ball_possession_time": self.ball_possession_blue,
                        "opponent_possession_time": self.ball_possession_yellow
                    })
                
                infos[agent_id].update({
                    "distance_to_ball": dist_to_ball,
                    "ball_touched": self.ball_touched,
                    "continuity_metrics": self.continuity_metrics.copy()
                })
                
                # Condições de término específicas para cada tarefa
                if self.task_level == 0 and self.ball_touched:
                    dones[agent_id] = True
                    dones["__all__"] = True
                elif self.task_level == 1:
                    # Termina o episódio se mantiver posse por tempo suficiente
                    if self.ball_possession_blue >= 90:  # 3 segundos (30 fps * 3)
                        dones[agent_id] = True
                        dones["__all__"] = True
        
        return observations, rewards, dones, truncated, infos