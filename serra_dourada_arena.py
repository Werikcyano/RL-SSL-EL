import argparse
import os
import yaml
import time
import numpy as np
from datetime import datetime
import json

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray import tune
import torch

from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest_blue, TorchBetaTest_yellow

def create_env(config):
    return SSLMultiAgentEnv(**config)

class SerraDouradaArena:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Configura o dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsando dispositivo: {self.device}")
        
        # Inicializa o Ray
        ray.init()
        
        # Registra os modelos customizados
        from ray.rllib.models import ModelCatalog
        ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
        ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
        ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
        
        # Registra o ambiente
        register_env("Soccer", lambda config: create_env(config))
        
        # Cria o ambiente
        self.env = SSLMultiAgentEnv(**self.config["env"])
        
        # Carrega os modelos
        self.blue_agent = self._load_agent("blue")
        self.yellow_agent = self._load_agent("yellow")
        
        # Inicializa contadores
        self.blue_score = 0
        self.yellow_score = 0
        self.match_time = self.config["arena"]["match_time"]
        self.speed_up = self.config["arena"]["speed_up"]
        
        # Determina o número de robôs de cada time
        self.num_blue = len(self.config["env"]["init_pos"]["blue"])
        self.num_yellow = len(self.config["env"]["init_pos"]["yellow"])
        print(f"Número de robôs: Azul = {self.num_blue}, Amarelo = {self.num_yellow}")
        
    def _load_agent(self, team):
        checkpoint_path = self.config["arena"]["checkpoints"][team]
        
        # Configuração básica do agente
        agent_config = {
            "env": "Soccer",
            "framework": "torch",
            "model": {
                "custom_model": "custom_vf_model",
                "custom_action_dist": f"beta_dist_{team}"
            },
            "env_config": self.config["env"],  # Adicionando a configuração do ambiente
            "num_gpus": 1 if torch.cuda.is_available() else 0,  # Configura GPU se disponível
        }
        
        # Carrega o agente do checkpoint
        agent = PPO(config=agent_config)
        agent.restore(checkpoint_path)
        return agent
    
    def _process_observation(self, obs):
        """Processa as observações para garantir que estejam no dispositivo correto"""
        if isinstance(obs, dict):
            return {k: self._process_observation(v) for k, v in obs.items()}
        elif isinstance(obs, np.ndarray):
            return torch.FloatTensor(obs).to(self.device)
        return obs
    
    def _get_actions(self, obs, team="blue"):
        """Obtém ações para todos os robôs de um time"""
        agent = self.blue_agent if team == "blue" else self.yellow_agent
        num_robots = self.num_blue if team == "blue" else self.num_yellow
        
        actions = {}
        for i in range(num_robots):
            agent_id = f"{team}_{i}"
            if agent_id in obs:
                # Converte o tensor para CPU e depois para numpy antes de passar para o agente
                if isinstance(obs[agent_id], torch.Tensor):
                    obs_np = obs[agent_id].cpu().numpy()
                else:
                    obs_np = obs[agent_id]
                
                # Obtém a ação do agente
                action = agent.compute_single_action(obs_np)
                if isinstance(action, tuple):
                    action = action[0]  # Pega apenas a ação, descarta extras
                
                # Garante que a ação seja um array numpy com 4 valores
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                
                # Aplica o sinal correto para cada time
                if team == "yellow":
                    action = np.array([-action[0], action[1], -action[2], action[3]])
                
                print(f"Ações para {agent_id}: {action} (tipo: {type(action)})")
                actions[agent_id] = action
            else:
                # Se não houver observação para este robô, use uma ação nula
                actions[agent_id] = np.zeros(4)  # 4 é o tamanho da ação (vx, vy, w, kick)
        
        return actions
    
    def run_match(self):
        print(f"\n=== Iniciando partida na Arena Serra Dourada ===")
        print(f"Time Azul vs Time Amarelo")
        print(f"Duração: {self.match_time/60:.1f} minutos")
        print(f"Fator de aceleração: {self.speed_up}x")
        
        obs = self.env.reset()
        start_time = time.time()
        elapsed_time = 0
        last_save_time = 0  # Tempo da última vez que salvamos as estatísticas
        save_interval = 10  # Intervalo em segundos para salvar estatísticas
        
        # Cria diretório para a partida atual
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        match_dir = f"/ws/arena_results/match_{timestamp}"
        os.makedirs(match_dir, exist_ok=True)
        
        # Inicializa dicionário para estatísticas
        match_stats = {
            'placar': {'azul': 0, 'amarelo': 0},
            'posse_de_bola': {'azul': 0, 'amarelo': 0},
            'chutes': {'azul': 0, 'amarelo': 0},
            'resets': {'total': 0, 'lateral': 0, 'endline': 0},
            'tempo_total': 0,
            'historico': []  # Lista para armazenar estatísticas ao longo do tempo
        }
        
        while elapsed_time < self.match_time:
            # Processa as observações
            processed_obs = self._process_observation(obs)
            
            # Obtém ações dos agentes
            blue_actions = self._get_actions(processed_obs, "blue")
            yellow_actions = self._get_actions(processed_obs, "yellow")
            
            # Combina as ações
            actions = {**blue_actions, **yellow_actions}
            
            # Executa um passo no ambiente
            obs, rewards, dones, truncated, infos = self.env.step(actions)
            
            # Renderiza o ambiente
            self.env.render()
            
            # Atualiza estatísticas se houver informações disponíveis
            if "blue_0" in infos and "score" in infos["blue_0"]:
                self.blue_score = infos["blue_0"]["score"]["blue"]
                self.yellow_score = infos["blue_0"]["score"]["yellow"]
                match_stats['placar']['azul'] = self.blue_score
                match_stats['placar']['amarelo'] = self.yellow_score
                
                # Atualiza métricas de continuidade se disponíveis
                if "continuity_metrics" in infos["blue_0"]:
                    metrics = infos["blue_0"]["continuity_metrics"]
                    match_stats['resets']['total'] = metrics.get('total_resets', 0)
                    match_stats['resets']['lateral'] = metrics.get('lateral_resets', 0)
                    match_stats['resets']['endline'] = metrics.get('endline_resets', 0)
                    match_stats['posse_de_bola']['azul'] = metrics.get('ball_possession_blue', 0)
                    match_stats['posse_de_bola']['amarelo'] = metrics.get('ball_possession_yellow', 0)
                
                # Mostra placar quando há alteração
                print(f"\rPlacar: Azul {self.blue_score} x {self.yellow_score} Amarelo | "
                      f"Tempo: {elapsed_time/60:.1f} min", end="")
            
            # Atualiza o tempo
            elapsed_time = (time.time() - start_time) * self.speed_up
            match_stats['tempo_total'] = elapsed_time
            
            # Salva estatísticas periodicamente
            if elapsed_time - last_save_time >= save_interval:
                # Adiciona snapshot atual ao histórico
                snapshot = {
                    'tempo': elapsed_time,
                    'placar': match_stats['placar'].copy(),
                    'posse_de_bola': match_stats['posse_de_bola'].copy(),
                    'resets': match_stats['resets'].copy()
                }
                match_stats['historico'].append(snapshot)
                
                # Salva arquivo temporário com estatísticas atuais
                temp_stats_file = f"{match_dir}/stats_temp.json"
                with open(temp_stats_file, 'w') as f:
                    json.dump(match_stats, f, indent=4, ensure_ascii=False)
                
                last_save_time = elapsed_time
            
            # Reset se necessário (bola fora, etc)
            if any(dones.values()) or any(truncated.values()):
                obs = self.env.reset()
        
        # Resultado final
        print("\n\n=== Fim de Jogo ===")
        print(f"Placar Final: Azul {self.blue_score} x {self.yellow_score} Amarelo")
        
        winner = "Azul" if self.blue_score > self.yellow_score else "Amarelo" if self.yellow_score > self.blue_score else "Empate"
        
        # Adiciona configurações e resultado final
        match_stats['vencedor'] = winner
        match_stats['config'] = {
            'duracao': self.match_time,
            'speed_up': self.speed_up,
            'num_robos': {
                'azul': self.num_blue,
                'amarelo': self.num_yellow
            }
        }
        
        # Salva arquivo final com todas as estatísticas
        final_stats_file = f"{match_dir}/stats_final.json"
        with open(final_stats_file, 'w') as f:
            json.dump(match_stats, f, indent=4, ensure_ascii=False)
            
        print(f"\nEstatísticas salvas em: {match_dir}")
        return winner
    
    def run_tournament(self):
        num_matches = self.config["arena"]["num_matches"]
        results = []
        
        print(f"\n=== Iniciando Torneio na Arena Serra Dourada ===")
        print(f"Número de partidas: {num_matches}")
        
        for match in range(num_matches):
            print(f"\nPartida {match + 1}/{num_matches}")
            match_result = self.run_match()
            results.append(match_result)
            
            # Reset do ambiente e contadores para próxima partida
            self.env.reset()
            self.blue_score = 0
            self.yellow_score = 0
        
        # Estatísticas do torneio
        blue_wins = sum(1 for r in results if r == "Azul")
        yellow_wins = sum(1 for r in results if r == "Amarelo")
        draws = sum(1 for r in results if r == "Empate")
        
        print("\n=== Resultados do Torneio ===")
        print(f"Vitórias do Azul: {blue_wins}")
        print(f"Vitórias do Amarelo: {yellow_wins}")
        print(f"Empates: {draws}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Arena Serra Dourada - Competição entre times treinados")
    parser.add_argument("--config", type=str, default="config_serra_dourada.yaml",
                      help="Caminho para o arquivo de configuração")
    args = parser.parse_args()
    
    arena = SerraDouradaArena(args.config)
    arena.run_tournament()

if __name__ == "__main__":
    main() 