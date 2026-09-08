# VitFly in Docker

This repo (`/home/wens/VitFly`) is bind-mounted into the container at
`/root/catkin_ws/src/vitfly`, so edits made here on the host are what the
container builds and runs — nothing is copied into the image. `build/` and
`devel/` live in a separate named Docker volume so catkin's build output
doesn't clutter this checkout.

## One-time host setup

Allow the container to open windows on your X server (needed each login session,
or add it to your shell profile):
```
xhost +local:docker
```

## Build and start

From the repo root:
```
docker compose build
docker compose up -d
docker compose exec vitfly bash
```

## First-time catkin build (inside the container)

The workspace is auto-initialized by the entrypoint. Just build it:
```
catkin build
source devel/setup.bash
cd src/vitfly
```

## Getting the required assets

`environments.tar`, `flightrender.tar`, `pretrained_models.tar`, and `data.zip`
are on a password-protected Box link (see main README, pw: `vitfly2025`) that
needs a logged-in browser — download them on the host into `downloads/`, then
from the repo root **on the host** (no Docker needed for this step):
```
./extract_assets.sh
```
It extracts each file to the exact path the upstream README specifies. Since
`downloads/` and the rest of the repo are bind-mounted, the extracted files are
immediately visible inside the container too.

For a first reproduction pass you only need `pretrained_models.tar`,
`environments.tar`, and `flightrender.tar` — skip `data.zip` (2.5GB, only
needed for training) until you actually want to retrain.

## Running the simulation eval

Inside the container, after building and extracting assets:
```
cd src/vitfly
bash launch_evaluation.bash 1 vision
```
RViz and the Unity render window should appear on your host desktop.

## Troubleshooting

- **rviz doesn't appear / crashes instantly (segfault, exit code -11), but the
  Unity render window does show up**: on a hybrid-GPU laptop (Intel iGPU +
  NVIDIA dGPU), nvidia-container-toolkit only passes through the reserved
  NVIDIA `/dev/dri` render node, not the Intel one the host X server actually
  uses for on-screen rendering. `docker-compose.yml` already mounts the full
  `/dev/dri` directory to fix this — if you hit this again, confirm with
  `ls /dev/dri` inside the container that both GPUs' `cardN`/`renderDN` nodes
  are present.
- **Nothing appears at all**: check `xhost +local:docker` was run on the host
  in this login session, and that `DISPLAY` is set before `docker compose up`.
- **Sim runs and reports results, but rviz/Unity windows never appear, and it
  feels instant / skips the ~10s boot messages**: `launch_evaluation.bash`
  only starts a fresh simulator `if [ -z $(pgrep visionsim_node) ]` — if a
  *previous* simulator process is still alive but its `rviz`/Unity windows
  were killed separately (e.g. a partial `pkill` that missed `rosmaster` /
  `visionsim_node`), every subsequent run silently reuses that orphaned,
  window-less, **imageless** session instead of launching a new one. Results
  from such a run are not meaningful — with no depth images being published,
  the model never receives real input. Check for this with:
  ```
  docker compose exec vitfly bash -lc "pgrep -a rosmaster; pgrep -a visionsim_node"
  docker compose exec vitfly bash -lc "source devel/setup.bash && rostopic hz /kingfisher/dodgeros_pilot/unity/depth"
  ```
  If `rostopic hz` reports no messages, the fix is a full container restart
  (`docker compose restart vitfly`) rather than trying to `pkill` every
  process by name — that guarantees no orphan survives.

## Notes

- GPU passthrough and `NVIDIA_DRIVER_CAPABILITIES=all` are already wired up in
  `docker-compose.yml` — `nvidia-smi` should work inside the container.
- `network_mode: host` is used so ROS master/node discovery and X11 both work
  without extra config; this container isn't meant to run alongside other
  ROS graphs on the same host.
- If the container was already built and you only changed Python files in
  `envtest/`, `models/`, or `training/`, no rebuild is needed — just re-run;
  only changes to `docker/Dockerfile` or `requirements-ml.txt` need
  `docker compose build` again.
