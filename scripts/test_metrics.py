import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.ssl_curriculum_env import SSLCurriculumEnv
import numpy as np
import yaml
import pprint

def create_force_out_action():
    """Cria uma ação que deve empurrar a bola para fora do campo"""
    return np.array([1.0, 0.0, 0.0, 1.0])  # Velocidade máxima em x e chute

def create_kick_action():
    """Cria uma ação apenas com chute"""
    return np.array([0.0, 0.0, 0.0, 1.0])  # Apenas chute

def print_state(env, step):
    """Imprime o estado atual do ambiente"""
    print(f"Step {step}:")
    print(f"Ball pos: x={env.frame.ball.x:.2f}, y={env.frame.ball.y:.2f}")
    print(f"Ball vel: vx={env.frame.ball.v_x:.2f}, vy={env.frame.ball.v_y:.2f}")
    print("Blue robots:")
    for i in range(env.n_robots_blue):
        robot = env.frame.robots_blue[i]
        print(f"  Robot {i}:")
        print(f"    pos: x={robot.x:.2f}, y={robot.y:.2f}, theta={robot.theta:.2f}")
        print(f"    vel: vx={robot.v_x:.2f}, vy={robot.v_y:.2f}, vtheta={robot.v_theta:.2f}")
        print(f"    wheels: w0={robot.v_wheel0:.2f}, w1={robot.v_wheel1:.2f}, w2={robot.v_wheel2:.2f}, w3={robot.v_wheel3:.2f}")
        print(f"    infrared: {robot.infrared}")

def print_commands(commands):
    print("Commands:")
    for cmd in commands:
        if not cmd.yellow:
            print(f"  Robot {cmd.id}:")
            print(f"    wheel_speed: {cmd.wheel_speed}")
            print(f"    v_x: {cmd.v_x}, v_y: {cmd.v_y}, v_theta: {cmd.v_theta}")
            print(f"    kick_v_x: {cmd.kick_v_x}, kick_v_z: {cmd.kick_v_z}")
            print(f"    dribbler: {cmd.dribbler}")

def test_metrics():
    # Carrega configurações
    with open("/ws/config.yaml") as f:
        configs = yaml.safe_load(f)
    
    # Configura o ambiente
    env_config = {
        **configs["env"],
        "curriculum_config": configs["curriculum"]
    }
    
    # Cria o ambiente
    env = SSLCurriculumEnv(**env_config)
    
    print("Iniciando teste das métricas de continuidade...")
    print("\nConfigurações do ambiente:")
    print(f"FPS: {env.fps}")
    print(f"Match Time: {env.max_ep_length/env.fps}s")
    print(f"Velocidade máxima: {env.max_v}")
    print(f"Velocidade angular máxima: {env.max_w}")
    print(f"Velocidade de chute: {env.kick_speed_x}")
    
    # Executa alguns episódios
    for episode in range(3):
        print(f"\nEpisódio {episode + 1}")
        obs, _ = env.reset()
        done = {"__all__": False}
        truncated = {"__all__": False}
        episode_steps = 0
        kick_steps = 0
        
        # Imprime estado inicial
        print("\nEstado Inicial:")
        print_state(env, 0)
        
        while not (done["__all__"] or truncated["__all__"]):
            # Testa diferentes padrões de chute
            if episode_steps < 100:  # Primeiros 100 steps: apenas chute
                actions = {
                    agent_id: create_kick_action()
                    for agent_id in env._agent_ids
                }
                kick_steps += 1
            elif episode_steps < 200:  # Próximos 100 steps: movimento + chute
                actions = {
                    agent_id: create_force_out_action()
                    for agent_id in env._agent_ids
                }
                kick_steps += 1
            else:  # Resto do tempo: ações aleatórias
                actions = {
                    agent_id: env.action_space[agent_id].sample()
                    for agent_id in env._agent_ids
                }
            
            obs, reward, done, truncated, info = env.step(actions)
            episode_steps += 1
            
            # Log do estado a cada 10 steps durante os primeiros 200 steps
            if episode_steps <= 200 and episode_steps % 10 == 0:
                print_state(env, episode_steps)
                print_commands(env.sent_commands)
            # Depois, log a cada 100 steps
            elif episode_steps > 200 and episode_steps % 100 == 0:
                print_state(env, episode_steps)
                print_commands(env.sent_commands)
            
            # Verifica se houve reset neste step
            metrics = info["blue_0"].get("continuity_metrics", {})
            if metrics:
                total_resets = metrics.get("total_resets", 0)
                if total_resets > 0:
                    print("\nMétricas de Continuidade Atuais:")
                    print(f"Total de Resets: {metrics['total_resets']}")
                    print(f"Resets Laterais: {metrics['lateral_resets']}")
                    print(f"Resets Endline: {metrics['endline_resets']}")
                    if metrics['time_between_resets']:
                        print(f"Média de Tempo entre Resets: {np.mean(metrics['time_between_resets']):.2f}s")
                    print(f"Sequência Atual sem Reset: {metrics['current_sequence_time']:.2f}s")
                    print(f"Maior Sequência sem Reset: {metrics['max_sequence_without_reset']:.2f}s")
        
        print(f"\nFim do Episódio {episode + 1}")
        print(f"Passos totais: {episode_steps}")
        print(f"Passos com chute: {kick_steps}")
        
    env.close()
    print("\nTeste concluído!")

if __name__ == "__main__":
    test_metrics() 