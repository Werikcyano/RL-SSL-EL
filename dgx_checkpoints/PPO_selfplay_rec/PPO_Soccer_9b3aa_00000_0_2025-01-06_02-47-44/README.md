# Experimento PPO_Soccer_9b3aa_00000_0_2025-01-06_02-47-44

## Objetivo do Experimento

Este experimento é uma continuação do treinamento normal por mais 6 horas e 30 minutos, utilizando as configurações definidas no arquivo `config.yaml`.

## Configurações do Experimento

As configurações utilizadas neste experimento são idênticas às definidas no arquivo `config.yaml`:

### Configurações Gerais
- **Score Average Over**: 100 # Número de episódios usados para calcular a média da pontuação
- **Timesteps Total**: 160.000.000 # Limite máximo de passos de tempo para todo o treinamento
- **Checkpoint Frequency**: 50 # Frequência em que os checkpoints são salvos durante o treinamento
- **Checkpoint Restore**: `/root/ray_results/PPO_selfplay_rec/PPO_Soccer_e7341_00000_0_2025-01-04_18-44-17/checkpoint_000005` # Caminho para restaurar um checkpoint específico

### Configurações RLlib
- **CPUs**: 7 # Número de CPUs utilizadas no treinamento
- **GPUs**: 1 # Número de GPUs utilizadas no treinamento
- **Workers**: 6 # Número de processos paralelos executando o ambiente
- **Envs per Worker**: 2 # Número de ambientes por worker para coleta paralela de experiências
- **Framework**: PyTorch # Framework de deep learning utilizado
- **Environment Checking**: Desabilitado # Desativa verificações de segurança do ambiente para melhor performance

### Configurações PPO
- **Batch Mode**: truncate_episodes # Modo de coleta de dados - trunca episódios no tamanho do fragmento
- **Rollout Fragment Length**: auto # Tamanho automático dos fragmentos de rollout
- **Train Batch Size**: 38.520 # Tamanho do lote de treinamento
- **SGD Minibatch Size**: 12.840 # Tamanho do mini-lote para otimização
- **Gamma**: 0,99 # Fator de desconto para recompensas futuras
- **Lambda**: 0,95 # Parâmetro lambda do GAE (Generalized Advantage Estimation)
- **Entropy Coefficient**: 0,01 # Coeficiente para incentivar exploração
- **KL Coefficient**: 0,0 # Coeficiente para penalização de divergência KL
- **Learning Rate**: 0,0004 # Taxa de aprendizado do otimizador
- **VF Loss Coefficient**: 0,5 # Peso da função valor na função de perda
- **Gradient Clip**: 0,5 # Limite para clipar gradientes e evitar explosão
- **SGD Iterations**: 5 # Número de iterações de otimização por batch
- **Clip Parameter**: 0,2 # Parâmetro epsilon para clipar a razão de probabilidade
- **VF Clip Parameter**: 100.000,0 # Valor para clipar a função valor
- **Action Normalization**: Desabilitado # Normalização do espaço de ações desativada

### Configurações de Avaliação
- **Interval**: 1 # Frequência de avaliação durante treinamento
- **Workers**: 0 # Número de workers dedicados à avaliação
- **Duration**: 1 episódio # Duração da avaliação
- **Environment**: Soccer_recorder # Ambiente usado para avaliação
- **Envs per Worker**: 1 # Ambientes por worker durante avaliação

### Modelo Personalizado
- **Hidden Layers**: [300, 200, 100] # Arquitetura das camadas ocultas da rede neural
- **VF Share Layers**: false # Indica se a função valor compartilha camadas com a política

### Configurações do Ambiente
- **FPS**: 30 # Frames por segundo da simulação
- **Match Duration**: 40 segundos # Duração de cada partida
- **Render Mode**: human # Modo de renderização do ambiente
- **Field Type**: 0 # Tipo de campo utilizado

#### Posições Iniciais
**Time Azul**:
- Jogador 1: [-1.5, 0.0, 0.0] # Posição inicial [x, y, ângulo] do jogador 1 azul
- Jogador 2: [-2.0, 1.0, 0.0] # Posição inicial [x, y, ângulo] do jogador 2 azul
- Jogador 3: [-2.0, -1.0, 0.0] # Posição inicial [x, y, ângulo] do jogador 3 azul

**Time Amarelo**:
- Jogador 1: [1.5, 0.0, 180.0] # Posição inicial [x, y, ângulo] do jogador 1 amarelo
- Jogador 2: [2.0, 1.0, 180.0] # Posição inicial [x, y, ângulo] do jogador 2 amarelo
- Jogador 3: [2.0, -1.0, 180.0] # Posição inicial [x, y, ângulo] do jogador 3 amarelo

**Bola**: [0, 0] # Posição inicial [x, y] da bola

### Configurações de Curriculum Learning
- **Enabled**: true # Habilita o aprendizado curricular
- **Initial Task**: 0 # Começa com a tarefa mais básica
- **Promotion Threshold**: 0.8 # Taxa de sucesso necessária para avançar
- **Evaluation Window**: 100 # Número de episódios para avaliar o desempenho

#### Tarefas do Curriculum
**Tarefa 0 (Básica)**:
- Máximo de passos: 300
- Distância de sucesso: 0.2

**Tarefa 1 (Intermediária)**:
- Máximo de passos: 400
- Distância de sucesso: 0.2

**Tarefa 2 (Avançada)**:
- Máximo de passos: 500
- Distância de sucesso: 0.2
