# Ajustes para Fase 0 do Treinamento Curriculum

## Problema Inicial
O ambiente apresentava inconsistências na Fase 0 do treinamento curriculum, onde robôs amarelos apareciam indevidamente e havia problemas com o mapeamento de políticas.

## Alterações Realizadas

### 1. Configuração do Ambiente (config.yaml)
- Modificação da configuração inicial para a Fase 0:
  ```yaml
  init_pos:
    blue:
      1: [-0.5,  0.0,    0.0]  # Atacante
      2: [-1.0,  0.0,    0.0]  # Intermediário
      3: [-2.0,  0.0,    0.0]  # Goleiro
    yellow: {}  # Nível 0 não tem robôs amarelos
    ball: [0, 0]
  ```

### 2. Ajuste do Sistema de Curriculum (rllib_multiagent.py)
- Implementação de uma função `create_curriculum_env` que:
  - Remove corretamente a configuração do curriculum antes de passar para o ambiente
  - Ajusta o número de agentes baseado no nível atual do curriculum
  - Garante que as posições iniciais sejam consistentes com o número de agentes

### 3. Mapeamento de Políticas
- Atualização da função `create_policy_mapping_fn` para:
  - Verificar corretamente o nível atual do curriculum
  - Retornar a política apropriada baseada na presença ou ausência de agentes amarelos
  - Garantir consistência entre treinamento e avaliação

### 4. Configuração de Avaliação
- Ajuste da configuração de avaliação para corresponder ao nível atual do curriculum:
  - Mesmo número de agentes
  - Mesmas posições iniciais
  - Política de mapeamento consistente

## Resultados
- Ambiente inicializa corretamente apenas com 3 robôs azuis na Fase 0
- Não há mais aparecimento indevido de robôs amarelos
- Sistema de recompensas funciona adequadamente para os agentes presentes
- Avaliação é executada de forma consistente com o estado atual do treinamento

## Próximos Passos
1. Monitorar o progresso do treinamento na Fase 0
2. Avaliar os critérios de promoção para a próxima fase
3. Verificar a qualidade das políticas aprendidas antes de avançar para a Fase 1 