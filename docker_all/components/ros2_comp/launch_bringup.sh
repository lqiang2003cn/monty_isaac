#!/usr/bin/env bash
# Launch bringup with use_moveit from USE_MOVEIT env (for real robot experiments).
# When USE_MOVEIT=1 or USE_MOVEIT=true, MoveIt + planner are started.
set -e
use_moveit="false"
[[ -n "$USE_MOVEIT" && "$USE_MOVEIT" != "0" ]] && use_moveit="true"
exec ros2 launch monty_demo opus_x3plus_bringup.launch.py use_moveit:=$use_moveit "$@"
