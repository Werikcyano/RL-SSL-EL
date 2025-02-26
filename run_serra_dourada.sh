#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Nome do container e imagem
CONTAINER_NAME="serra_dourada_arena"
IMAGE_NAME="pequi-ssl-rl-serra-dourada"  # Nome modificado da imagem

# Criar diretório para estatísticas se não existir
mkdir -p arena_results

# Configuração do X11
xhost +local:docker

# Obtém o ID do usuário atual
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

echo -e "${YELLOW}Parando e removendo container anterior...${NC}"
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1

# Construir a imagem
echo -e "${YELLOW}Construindo imagem da Arena Serra Dourada...${NC}"
docker build -t $IMAGE_NAME -f Dockerfile.serra_dourada .

echo -e "${YELLOW}Iniciando container da Arena Serra Dourada...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTH \
    -e QT_X11_NO_MITSHM=1 \
    -v $XSOCK:$XSOCK:rw \
    -v $XAUTH:$XAUTH:rw \
    -v $(pwd)/videos:/ws/videos \
    -v $(pwd)/dgx_checkpoints:/root/ray_results \
    -v $(pwd)/arena_results:/ws/arena_results \
    --device /dev/dri:/dev/dri \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-modeset:/dev/nvidia-modeset \
    -v /dev/shm:/dev/shm \
    --privileged \
    --shm-size=10gb \
    --network host \
    $IMAGE_NAME tail -f /dev/null

# Executa a arena
echo -e "${YELLOW}Iniciando Arena Serra Dourada...${NC}"
docker exec -it $CONTAINER_NAME python serra_dourada_arena.py "$@"

# Limpa a permissão do X11
xhost -local:docker

echo -e "${GREEN}Processo completo!${NC}" 