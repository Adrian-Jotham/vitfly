#!/bin/bash
# RGB -> Depth Anything V2 -> ViTLSTM variant of launch_evaluation.bash.
# Same simulator/trial harness as the original, but drives the policy from
# envtest/ros/run_competition_rgb.py (monocular depth estimate) instead of
# Flightmare's ground-truth depth topic. See models/DepthAnythingV2/ for the
# depth estimator and its empirical calibration.
#
# Usage: bash launch_evaluation_rgb.bash <N trials>

if [ $1 ]
then
  N="$1"
else
  N=5
fi

if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 10
else
  ROS_PID=""
fi

SUMMARY_FILE="evaluation_rgb.yaml"
echo "" > $SUMMARY_FILE

datetime=$(date '+d%m_%d_t%H_%M')

for i in $(eval echo {1..$N})
do
  start_time=$(date +%s)

  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py ${datetime}_rgb_N$i &
  PY_PID="$!"

  python3 run_competition_rgb.py --des_vel 5.0 --model_type "ViTLSTM" --model_path ../../models/ViTLSTM_model.pth &
  COMP_PID="$!"

  cd -

  sleep 2

  while ps -p $PY_PID > /dev/null
  do
    echo
    echo [LAUNCH_EVALUATION_RGB] Sending start navigation command
    echo
    rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
    sleep 2

    if ((($(date +%s) - start_time) >= 300))
    then
      echo "Time limit exceeded. Exiting evaluation script loop."
      kill -SIGINT $PY_PID
      break
    fi
  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp_rgb.yaml"
  mv "tmp_rgb.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
