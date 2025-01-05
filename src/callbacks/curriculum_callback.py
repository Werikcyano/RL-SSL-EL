from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

class CurriculumCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.success_history = []
        self.current_task = 0
        
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
    # Verifica se o episódio foi bem-sucedido (robô chegou na bola)
        try:
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
        except Exception as e:
            print(f"Erro no callback: {e}")