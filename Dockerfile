# Descrição: Dockerfile para criar um contêiner com o ambiente de desenvolvimento do rSim

# Use uma imagem base Python oficial
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do contêiner (ws == workspace)
WORKDIR /ws

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    libode-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências do Python listadas em requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install rSim
RUN pip install git+https://github.com/Pequi-Mecanico-SSL/rSim.git

# Copy the rSoccer directory
RUN mkdir /rsoccer_gym
COPY rsoccer_gym rsoccer_gym

COPY rllib_multiagent.py .
COPY action_dists.py .
COPY custom_torch_model.py .
COPY config.yaml .

RUN mkdir /ws/volume
RUN mkdir /ws/scripts

# Iniciar o contêiner com o bash
CMD ["/bin/bash"]