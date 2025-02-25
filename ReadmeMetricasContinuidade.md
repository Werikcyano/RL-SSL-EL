# Métricas de Continuidade para Avaliação do Comportamento do Robô

## Visão Geral
Este documento descreve as métricas de continuidade implementadas para avaliar o comportamento dos robôs no simulador SSL. As métricas foram projetadas para monitorar a estabilidade e eficiência do movimento dos robôs, com foco especial em eventos de reset e a capacidade de manter operação contínua.

## Métricas Implementadas

### 1. Contadores de Reset
- **Total de Resets**: Número total de resets ocorridos durante um episódio
- **Resets Laterais**: Número de resets causados por saídas laterais da bola
- **Resets por Linha de Fundo**: Número de resets causados por saídas pela linha de fundo

### 2. Métricas Temporais
- **Tempo Entre Resets**: Lista dos intervalos de tempo entre resets consecutivos
- **Sequência Atual sem Reset**: Tempo decorrido desde o último reset
- **Maior Sequência sem Reset**: Maior intervalo de tempo alcançado sem reset

## Implementação

### Classe SSLCurriculumEnv
```python
self.continuity_metrics = {
    'total_resets': 0,
    'lateral_resets': 0,
    'endline_resets': 0,
    'time_between_resets': [],
    'last_reset_time': None,
    'current_sequence_time': 0,
    'max_sequence_without_reset': 0
}
```

### Método track_reset
O método `track_reset` é chamado sempre que ocorre um reset, seja por saída lateral ou pela linha de fundo:
- Incrementa os contadores apropriados
- Calcula e registra o tempo entre resets
- Atualiza a sequência máxima sem reset
- Reinicia o contador de sequência atual

### Callback de Curriculum
O `CurriculumCallback` processa e agrega as métricas para análise:
- Calcula médias de resets por episódio
- Monitora tempos médios entre resets
- Acompanha sequências máximas sem reset
- Calcula distribuição dos tipos de reset

## Uso das Métricas

### 1. Avaliação de Desempenho
As métricas permitem avaliar:
- Estabilidade do robô durante o movimento
- Eficiência no controle da bola
- Capacidade de evitar saídas desnecessárias

### 2. Treinamento do Agente
Auxiliam no processo de treinamento:
- Identificação de comportamentos problemáticos
- Ajuste de parâmetros de recompensa
- Avaliação de progresso entre diferentes níveis de currículo

### 3. Análise de Progresso
Fornecem insights sobre:
- Evolução do aprendizado
- Pontos de melhoria necessários
- Comparação entre diferentes versões do agente

## Integração com o Sistema de Recompensas
As métricas podem ser utilizadas para:
- Ajustar recompensas baseadas na continuidade do movimento
- Penalizar resets frequentes
- Recompensar sequências longas sem reset

## Visualização e Monitoramento
As métricas são registradas e podem ser visualizadas:
- Durante o treinamento através dos logs
- Em tempo real durante a execução
- Em relatórios de análise pós-treinamento

## Considerações Técnicas

### Cálculo de Tempo
- Tempo é medido em steps da simulação
- Conversão para segundos: `steps / fps`
- Precisão temporal baseada no timestep da simulação

### Tipos de Reset
1. **Reset Lateral**
   - Ocorre quando a bola sai pelas laterais do campo
   - Registrado como `lateral_reset`

2. **Reset por Linha de Fundo**
   - Ocorre quando a bola sai pelas linhas de fundo
   - Registrado como `endline_reset`

### Armazenamento de Dados
- Métricas são mantidas em dicionário durante execução
- Dados podem ser exportados para análise posterior
- Integração com sistema de logging do ambiente

## Exemplo de Uso

```python
# Acessando métricas durante treinamento
metrics = info["blue_0"].get("continuity_metrics", {})
if metrics:
    total_resets = metrics['total_resets']
    current_sequence = metrics['current_sequence_time']
    max_sequence = metrics['max_sequence_without_reset']
``` 