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

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):
        episode.hist_data["score"] = []
        # Obtém o nível atual do curriculum
        if hasattr(worker.env, "task_level"):
            self.task_level = worker.env.task_level
            print(f"\n[TREINAMENTO] Task Atual: {self.task_level}")

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
                    "goals_per_episode": metrics['goals_per_episode'],
                    "current_task_level": self.task_level
                })
            
            # Processa métricas de score
            info_a = episode.last_info_for("blue_0")
            single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]
            self.success_rate.append(1 if single_score > 0 else 0)

            # Verifica se é hora de promover para o próximo nível
            if len(self.success_rate) >= 100:
                success_rate = sum(self.success_rate) / len(self.success_rate)
                print(f"\n[CURRICULUM] Taxa de Sucesso Atual: {success_rate:.2f}")
                
                if success_rate >= 0.8 and hasattr(worker.env, "task_level"):
                    worker.env.task_level = min(worker.env.task_level + 1, 2)
                    print(f"[CURRICULUM] Promovido para Task {worker.env.task_level}!")
                    self.success_rate.clear()
                
            # Processa métricas de sucesso (código existente)
            final_distance = episode.last_info_for("blue_0").get("distance_to_ball")
            if final_distance is None:
                print("Aviso: distance_to_ball não encontrado no info")
                return
                
            success = final_distance < worker.config["env_config"]["curriculum_config"]["tasks"][self.task_level]["success_distance"]
            
            # Calcula taxa de sucesso
            success_rate = np.mean(self.success_rate)
            
            # Verifica se deve avançar para próxima tarefa
            if (success_rate >= worker.config["env_config"]["curriculum_config"]["promotion_threshold"] and 
                self.task_level < 2):
                self.task_level += 1
                print(f"Avançando para tarefa {self.task_level}")
                
                # Atualiza o nível da tarefa em todos os ambientes
                for env in base_env.get_unwrapped():
                    env.task_level = self.task_level
                    
                # Reseta estatísticas de continuidade ao mudar de nível
                self.continuity_stats = {
                    'episode_resets': [],
                    'avg_time_between_resets': [],
                    'max_sequence_lengths': [],
                    'reset_distribution': {
                        'lateral': [],
                        'endline': []
                    }
                }
                
        except Exception as e:
            print(f"Erro no callback: {e}")
            
    def get_continuity_statistics(self) -> Dict:
        """Retorna estatísticas agregadas das métricas de continuidade"""
        if not self.continuity_stats['episode_resets']:
            return {}
            
        return {
            'mean_resets_per_episode': np.mean(self.continuity_stats['episode_resets']),
            'mean_time_between_resets': np.mean(self.continuity_stats['avg_time_between_resets']) if self.continuity_stats['avg_time_between_resets'] else 0,
            'max_sequence_length': max(self.continuity_stats['max_sequence_lengths']) if self.continuity_stats['max_sequence_lengths'] else 0,
            'lateral_reset_ratio': np.mean(self.continuity_stats['reset_distribution']['lateral']) / max(np.mean(self.continuity_stats['episode_resets']), 1),
            'endline_reset_ratio': np.mean(self.continuity_stats['reset_distribution']['endline']) / max(np.mean(self.continuity_stats['episode_resets']), 1)
        }