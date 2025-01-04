# Experimento PPO_Soccer_e7341_00000_0_2025-01-04_18-44-17

Este experimento foi executado utilizando a versão padrão do framework de treinamento. O último update antes de parar a execução foi:

╭─────────────────────────────────────────────────────╮
│ Trial PPO_Soccer_e7341_00000 result                 │
├─────────────────────────────────────────────────────┤
│ episodes_total                                57534 │
│ num_env_steps_sampled                   1.23264e+07 │
│ num_env_steps_trained                   1.23264e+07 │
│ sampler_results/episode_len_mean            149.477 │
│ sampler_results/episode_reward_mean        -148.264 │
╰─────────────────────────────────────────────────────╯

As configurações utilizadas neste experimento estão definidas no arquivo `config.yaml`:

score_average_over: 100
timesteps_total: 160000000 # Número máximo de timesteps para o treinamento
checkpoint_freq: 50 # Frequência de salvamento dos checkpoints (a cada 50 iterações)
checkpoint_restore: "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_28842_00000_0_2024-12-06_02-52-40/checkpoint_000007" # Caminho para restaurar um checkpoint anterior
rllib:
  num_cpus: 7  # Número de CPUs utilizadas (deve ser igual ou maior que num_workers)
  num_gpus: 1 # Número de GPUs utilizadas
  num_workers: 6  # Número de workers (ambientes em paralelo)
  num_envs_per_worker: 2 # Número de ambientes por worker
  framework: "torch" # Framework utilizado (PyTorch)
  disable_env_checking: true # Desabilita a verificação do ambiente
PPO:
  batch_mode: "truncate_episodes"
  rollout_fragment_length: "auto"
  train_batch_size: 38520 # Tamanho do batch de treinamento (workers * envs * fragment)
  sgd_minibatch_size: 12840 # Tamanho do mini-batch SGD (batch / 3)
  gamma: 0.99 # Fator de desconto
  lambda: 0.95 # Parâmetro lambda para o Generalized Advantage Estimation (GAE)
  entropy_coeff: 0.01 # Coeficiente de entropia
  kl_coeff: 0.0 # Coeficiente KL
  lr: 0.0004 # Taxa de aprendizado
  vf_loss_coeff: 0.5 # Coeficiente de perda da função valor
  grad_clip: 0.5 # Limite para o gradiente (ajuda a evitar NaNs nos pesos da rede)
  num_sgd_iter: 5 # Número de iterações SGD
  clip_param: 0.2 # Parâmetro de clipping do PPO
  vf_clip_param: 100000.0 # Parâmetro de clipping da função valor (essencialmente desativado)
  normalize_actions: false # Não normaliza as ações

evaluation:
  evaluation_interval: 1 # Intervalo de avaliação (a cada 1 iteração)
  evaluation_num_workers: 0 # Número de workers para avaliação
  evaluation_duration: 1 # Duração da avaliação
  evaluation_duration_unit: "episodes" # Unidade da duração da avaliação (episódios)
  evaluation_config:
    env: "Soccer_recorder" # Ambiente utilizado para avaliação
    num_envs_per_worker: 1 # Número de ambientes por worker na avaliação
  
custom_model:
  fcnet_hiddens: [300, 200, 100] # Camadas ocultas da rede neural personalizada
  vf_share_layers: false # Não compartilha camadas entre a política e a função valor
env:
  init_pos: # Posições iniciais dos jogadores
    blue:
      1: [-1.5,  0.0,    0.0]
      2: [-2.0,  1.0,    0.0]
      3: [-2.0, -1.0,    0.0]
    yellow:
      1: [ 1.5,  0.0,  180.0]
      2: [ 2.0,  1.0,  180.0]
      3: [ 2.0, -1.0,  180.0]
    ball: [0, 0] # Posição inicial da bola
  field_type: 0 # Tipo de campo
  fps: 30 # Frames por segundo
  match_time: 40 # Duração da partida em segundos
  render_mode: "human" # Modo de renderização



