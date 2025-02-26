from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
from typing import Dict
from collections import deque
from ray.rllib.evaluation.episode import Episode

class CurriculumCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.task_level = 0
        self.success_rate = deque(maxlen=100)
        self.continuity_stats = {
            'episode_resets': [],
            'avg_time_between_resets': [],
            'max_sequence_lengths': [],
            'reset_distribution': {
                'lateral': [],
                'endline': []
            }
        }
        # Métricas específicas para Tarefa 1
        self.task1_stats = {
            'possession_time': [],
            'opponent_possession_time': [],
            'successful_possessions': 0,
            'total_episodes': 0
        }

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):
        episode.hist_data["score"] = []
        # Obtém o nível atual do curriculum
        if hasattr(worker.env, "task_level"):
            self.task_level = worker.env.task_level
            print(f"\n[TREINAMENTO] Task Atual: {self.task_level}")
            
            # Inicializa contadores específicos para Tarefa 1
            if self.task_level == 1:
                self.task1_stats['total_episodes'] += 1

    def on_episode_end(
        self, *, worker, base_env, policies, episode: Episode, **kwargs
    ) -> None:
        try:
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
                
                # Processa métricas específicas da Tarefa 1
                if self.task_level == 1:
                    info = episode.last_info_for("blue_0")
                    if "ball_possession_time" in info:
                        self.task1_stats['possession_time'].append(info["ball_possession_time"])
                    if "opponent_possession_time" in info:
                        self.task1_stats['opponent_possession_time'].append(info["opponent_possession_time"])
                    
                    # Verifica se foi uma posse bem-sucedida (3 segundos = 90 frames)
                    if info.get("ball_possession_time", 0) >= 90:
                        self.task1_stats['successful_possessions'] += 1
                
                # Adiciona métricas customizadas ao episódio
                episode.custom_metrics.update({
                    "total_resets": metrics['total_resets'],
                    "avg_time_between_resets": avg_time if metrics['time_between_resets'] else 0,
                    "max_sequence_without_reset": metrics['max_sequence_without_reset'],
                    "lateral_resets_ratio": metrics['lateral_resets'] / max(metrics['total_resets'], 1),
                    "endline_resets_ratio": metrics['endline_resets'] / max(metrics['total_resets'], 1),
                    "goals_blue": metrics.get('goals_blue', 0),
                    "goals_yellow": metrics.get('goals_yellow', 0),
                    "total_goals": metrics.get('total_goals', 0),
                    "goals_per_episode": metrics.get('goals_per_episode', 0),
                    "current_task_level": self.task_level
                })
                
                # Adiciona métricas específicas da Tarefa 1
                if self.task_level == 1:
                    possession_success_rate = self.task1_stats['successful_possessions'] / max(self.task1_stats['total_episodes'], 1)
                    avg_possession_time = np.mean(self.task1_stats['possession_time']) if self.task1_stats['possession_time'] else 0
                    
                    episode.custom_metrics.update({
                        "possession_success_rate": possession_success_rate,
                        "avg_possession_time": avg_possession_time,
                        "successful_possessions": self.task1_stats['successful_possessions'],
                        "total_task1_episodes": self.task1_stats['total_episodes']
                    })
            
            # Processa métricas de score e sucesso
            info_a = episode.last_info_for("blue_0")
            
            # Critérios de sucesso específicos para cada tarefa
            success = False
            if self.task_level == 0:
                # Tarefa 0: Sucesso se tocou na bola
                success = info_a.get("ball_touched", False)
            elif self.task_level == 1:
                # Tarefa 1: Sucesso se fez gol
                success = info_a.get("goals_blue", 0) > 0
                
                # Atualiza estatísticas de gols
                episode.custom_metrics.update({
                    "goals_blue": info_a.get("goals_blue", 0),
                    "goals_yellow": info_a.get("goals_yellow", 0),
                    "total_goals": info_a.get("total_goals", 0),
                    "goals_per_episode": info_a.get("goals_per_episode", 0),
                    "episodes_with_goals": 1 if info_a.get("goals_blue", 0) > 0 else 0,
                    "episodes_without_goals": 1 if info_a.get("goals_blue", 0) == 0 else 0,
                    "episodes_with_opponent_goals": 1 if info_a.get("goals_yellow", 0) > 0 else 0
                })
            else:
                # Tarefa 2: Mantém lógica existente
                final_distance = info_a.get("distance_to_ball")
                if final_distance is not None:
                    success = final_distance < worker.config["env_config"]["curriculum_config"]["tasks"][self.task_level]["success_distance"]
            
            self.success_rate.append(1 if success else 0)

            # Verifica se é hora de promover para o próximo nível
            if len(self.success_rate) >= 100:
                success_rate = sum(self.success_rate) / len(self.success_rate)
                print(f"\n[CURRICULUM] Taxa de Sucesso Atual: {success_rate:.2f}")
                
                if success_rate >= worker.config["env_config"]["curriculum_config"]["promotion_threshold"] and self.task_level < 2:
                    self.task_level += 1
                    print(f"[CURRICULUM] Promovido para Task {self.task_level}!")
                    
                    # Atualiza o nível da tarefa em todos os ambientes
                    for env in base_env.get_unwrapped():
                        env.task_level = self.task_level
                    
                    # Reseta estatísticas
                    self.success_rate.clear()
                    self.continuity_stats = {
                        'episode_resets': [],
                        'avg_time_between_resets': [],
                        'max_sequence_lengths': [],
                        'reset_distribution': {
                            'lateral': [],
                            'endline': []
                        }
                    }
                    if self.task_level == 1:
                        self.task1_stats = {
                            'possession_time': [],
                            'opponent_possession_time': [],
                            'successful_possessions': 0,
                            'total_episodes': 0
                        }
                
        except Exception as e:
            print(f"Erro no callback: {e}")
            
    def get_continuity_statistics(self) -> Dict:
        """Retorna estatísticas agregadas das métricas de continuidade"""
        if not self.continuity_stats['episode_resets']:
            return {}
            
        stats = {
            'mean_resets_per_episode': np.mean(self.continuity_stats['episode_resets']),
            'mean_time_between_resets': np.mean(self.continuity_stats['avg_time_between_resets']) if self.continuity_stats['avg_time_between_resets'] else 0,
            'max_sequence_length': max(self.continuity_stats['max_sequence_lengths']) if self.continuity_stats['max_sequence_lengths'] else 0,
            'lateral_reset_ratio': np.mean(self.continuity_stats['reset_distribution']['lateral']) / max(np.mean(self.continuity_stats['episode_resets']), 1),
            'endline_reset_ratio': np.mean(self.continuity_stats['reset_distribution']['endline']) / max(np.mean(self.continuity_stats['episode_resets']), 1)
        }
        
        # Adiciona estatísticas específicas da Tarefa 1 se estiver nela
        if self.task_level == 1:
            stats.update({
                'possession_success_rate': self.task1_stats['successful_possessions'] / max(self.task1_stats['total_episodes'], 1),
                'avg_possession_time': np.mean(self.task1_stats['possession_time']) if self.task1_stats['possession_time'] else 0,
                'total_successful_possessions': self.task1_stats['successful_possessions'],
                'total_task1_episodes': self.task1_stats['total_episodes']
            })
            
        return stats