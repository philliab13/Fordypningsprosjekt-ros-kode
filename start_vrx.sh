#!/usr/bin/env bash

set -e

WS="$HOME/Documents/vrx_ws"

echo "======================================"
echo " Building perception packages"
echo "======================================"

cd "$WS"

source /opt/ros/jazzy/setup.bash

colcon build \
    --merge-install \
    --packages-select stereo_perception lidar_free_space

source "$WS/install/setup.bash"

echo
echo "======================================"
echo " Starting VRX / Gazebo"
echo "======================================"

terminator \
    --new-tab \
    -T "VRX Gazebo" \
    -x bash -c "
        source /opt/ros/jazzy/setup.bash
        source '$WS/install/setup.bash'

        exec ros2 launch vrx_gz competition.launch.py \
            world:=gymkhana_task
    " &


echo "Waiting for Gazebo simulation clock..."

until ros2 topic list 2>/dev/null | grep -q "^/clock$"; do
    sleep 1
done

echo "Gazebo clock detected."

# Give the WAM-V and sensors a little time to spawn.
sleep 3


echo
echo "======================================"
echo " Starting RViz"
echo "======================================"

terminator \
    --new-tab \
    -T "VRX RViz" \
    -x bash -c "
        source /opt/ros/jazzy/setup.bash
        source '$WS/install/setup.bash'

        # Helper process:
        # wait for RViz to exist, then enable simulation time.
        (
            until ros2 node list 2>/dev/null | grep -q '^/rviz2$'; do
                sleep 0.5
            done

            echo 'RViz detected.'
            echo 'Enabling simulation time...'

            ros2 param set /rviz2 use_sim_time true

            echo 'RViz use_sim_time:'
            ros2 param get /rviz2 use_sim_time
        ) &

        # Keep the actual launch process in the foreground so
        # Ctrl+C is propagated properly.
        exec ros2 launch vrx_gazebo rviz.launch.py
    " &


# Estimate polar lidar free space in the Gazebo world frame.
terminator \
    --new-tab \
    -T "Lidar Free Space" \
    -x bash -c "
        source /opt/ros/jazzy/setup.bash
        source '$WS/install/setup.bash'
        exec ros2 launch lidar_free_space free_space.launch.py
    " &

# Give RViz a moment to start before opening stereo viewer.
sleep 2


echo
echo "======================================"
echo " Starting stereo perception"
echo "======================================"

terminator \
    --new-tab \
    -T "Stereo Perception" \
    -x bash -c "
        source /opt/ros/jazzy/setup.bash
        source '$WS/install/setup.bash'

        exec ros2 run stereo_perception stereo_viewer
    " &


echo
echo "======================================"
echo " VRX stack started"
echo "======================================"
echo
echo "Gazebo  : VRX simulation"
echo "RViz    : use_sim_time=true"
echo "Stereo  : stereo_perception/stereo_viewer"
echo "Lidar   : /lidar_free_space/polygons (RViz MarkerArray)"
echo
echo "Press Ctrl+C inside each Terminator tab to stop its process."