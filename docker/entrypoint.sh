#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash

cd /root/catkin_ws

if [ ! -d .catkin_tools ]; then
  catkin init
  catkin config --extend /opt/ros/noetic
  catkin config --merge-devel
  catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color
fi

# equivalent of the remaining, non-apt steps in setup_ros.bash
if [ -d src/vitfly ]; then
  touch src/vitfly/flightmare/flightros/CATKIN_IGNORE
  mkdir -p src/vitfly/envtest/ros/train_set
  export FLIGHTMARE_PATH=/root/catkin_ws/src/vitfly/flightmare
fi

if [ -f devel/setup.bash ]; then
  source devel/setup.bash
fi

exec "$@"
