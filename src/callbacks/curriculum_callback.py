from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
from typing import Dict

class CurriculumCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.success_history = []
        self.current_task = 0
        self.continuity_stats = {
            'episode_resets': [],
            'avg_time_between_resets': [],
            'max_sequence_lengths': [],
            'reset_distribution': {
                'lateral': [],
                'endline': []
            }
        }
        
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
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
                    "endline_resets_ratio": metrics['endline_resets'] / max(metrics['total_resets'], 1)
                })
            
            # Processa métricas de sucesso (código existente)
            final_distance = episode.last_info_for("blue_0").get("distance_to_ball")
            if final_distance is None:
                print("Aviso: distance_to_ball não encontrado no info")
                return
                
            success = final_distance < worker.config["env_config"]["curriculum_config"]["tasks"][self.current_task]["success_distance"]
            
            self.success_history.append(success)
            if len(self.success_history) > worker.config["env_config"]["curriculum_config"]["evaluation_window"]:
                self.success_history.pop(0)
                
            # Calcula taxa de sucesso
            success_rate = np.mean(self.success_history)
            
            # Verifica se deve avançar para próxima tarefa
            if (success_rate >= worker.config["env_config"]["curriculum_config"]["promotion_threshold"] and 
                self.current_task < 2):
                self.current_task += 1
                print(f"Avançando para tarefa {self.current_task}")
                
                # Atualiza o nível da tarefa em todos os ambientes
                for env in base_env.get_unwrapped():
                    env.task_level = self.current_task
                    
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