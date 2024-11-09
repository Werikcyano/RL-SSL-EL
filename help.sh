docker rm pequi-ssl
docker build -t pequi-ssl .
docker run --gpus all --name pequi-ssl -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it pequi-ssl python rllib_multiagent.py #render_episode.py