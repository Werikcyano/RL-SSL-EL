#!/bin/bash

# Nome do container e da imagem
CONTAINER_NAME="werikcyano_pequi_ssl_rl"
IMAGE_NAME="pequi-ssl-rl"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuração do X11
xhost +local:docker

echo -e "${YELLOW}Parando e removendo container anterior...${NC}"
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1

echo -e "${YELLOW}Construindo nova imagem...${NC}"
docker build -t $IMAGE_NAME .

echo -e "${YELLOW}Iniciando novo container...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/videos:/ws/videos \
    -v $(pwd)/dgx_checkpoints/PPO_selfplay_rec:/root/ray_results/PPO_selfplay_rec \
    --shm-size=10gb \
    $IMAGE_NAME tail -f /dev/null

echo -e "${YELLOW}Iniciando treinamento com curriculum e avaliação...${NC}"
docker exec -it $CONTAINER_NAME python rllib_multiagent.py 

# Limpa a permissão do X11
xhost -local:docker

echo -e "${GREEN}Processo completo!${NC}" 