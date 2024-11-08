Este repostório utiliza a biblioteca [rsoccer](https://github.com/robocin/rSoccer) para aplicar algortimos de Reinforcement Learning (RL) no ambinete Small Size League - EntryLevel (SSL). 

# Informações do ambiente
A implementação feita usa o conceito de self-play para que robôs aprendam a jogar, inspirado no trabalho [*Multiagent Reinforcement Learning for Strategic Decision Making and Control in Robotic Soccer Through Self-Play*](https://ieeexplore.ieee.org/document/9817118) no ambiente Very Small Size Soccer (VSSS).  O ambiente foi contruído pensando em um jogo 3x3.

## Episódio
Um episódio é finalizado assim que um gol é marcado ou atinja um tempo limite.

## Espaço de ações
Um robô pode possui 4 ações continuas: alterar velocidade no eixo x, alterar velocidade no eixo y, alterar velocidade ângular, chutar a bola.

## Espaço de observaçoes
Um robô a cada iteração com o ambiente coleta observações sobre ele. Essas informações coletadas são compostas pelas coordenadas dos robôs e da bola, distância e ângulos entre os aliados e adversários e tempo restante da partida. Assim, cada robô tem como obsevação um vetor de 77 valores.

## Recompensa
As recompensas são calculadas com base em 4 aspectos, 2 deles sendo compartilhados pelo time e os outros 2 individuais. Os compartilhados são velocidade da bola (r_speed) e distancia até a bola do robô aliado mais próximo da bola (r_dist). As inviduais medem o quão ofensiva e defensiva a posição do robô é no momento, a ofensiva (r_off) é o ângulo entre robô, bola e o gol do adversário, a defensiva (r_def) é o angulo entre gol aliado, o robô e a bola. Além disso, há também o componente relacionada a rotação do robô (v_theta), penalizando movimentos desnecessários. No fim, a composiçào da recompensa final é: 

- 30% da r_speed
- 40% da r_dist
- 5% da r_off
- 5% da r_def
- 20% da v_theta

# Rodando o código

**Clone o reposítorio com o comando:**

    git clone https://<seu-username>:<seu-token>@github.com/Pequi-Mecanico-SSL/RL.git

*Obs: Caso nào tenha, gere o seu token em: *Settings > Developer settings > Personal access tokens*.

**Construa a imagem:**

    Docker build -t ssl-el .

**Rode o container com volume:**

    docker run --gpus all --name pequi-ssl -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it pequi-ssl

**Para renderizar um episódio**
Em outro terminal fora do container rode:
    xhost +local:root
    
Dentro do container coloque:
    python render_episode.py

**Para treinar o modelo**

    python rllib_multiagent.py

Caso não esteja reconhecendo a gpu, tente instalar o [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) ou mudar a versão do cuda no dockerfile
    
# Personalizando hiperparâmetros

Para alterar os hiperparametros do algortimo, configurações de ambiente ou de treinamento, modifique o arquivo config.yaml. Depois, salve e contrua a imagem novamente.

