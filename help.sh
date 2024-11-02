docker rm pequi-ssl
docker build -t pequi-ssl .
docker run --gpus all --name pequi-ssl -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it pequi-ssl #python render_episode.py

# pc = pathlib.Path(checkpoint)
# f, oc = pyarrow.fs.FileSystem.from_uri(checkpoint)
# l = f.get_file_info(pyarrow.fs.FileSelector(pc.as_posix(), recursive=False))


# [
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/params.pkl': type=FileType.File, size=15277>, 
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/progress.csv': type=FileType.File, size=1050166>,
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/checkpoint-1': type=FileType.Directory>, 
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/events.out.tfevents.1730346003.aebccd88816e': type=FileType.File, size=1189439>, 
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/params.json': type=FileType.File, size=1913>, 
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/checkpoint_000000': type=FileType.Directory>, 
#     <FileInfo for '/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c6a18_00000_0_2024-10-31_03-39-47/result.json': type=FileType.File, size=6860637>]

#     _exists_at_fs_path(
#             filesystem, (checkpoint / "rllib_checkpoint.json").as_posix()
#         )
# f.get_file_info((pc/"rllib_checkpoint.json").as_posix())