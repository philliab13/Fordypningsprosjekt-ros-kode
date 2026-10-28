import os
from math import pi
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('nearest_obstacle_shoreline_detector'),
                          'config', 'shoreline.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=config),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # MA2's existing water frame has Z down. Keep its origin and flip Y/Z
        # for the shoreline node's camera-height and water-plane convention.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='shoreline_water_plane_up_tf', output='screen',
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', str(pi), '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'ma2_water_plane',
                        '--child-frame-id', 'ma2_water_plane_up'],
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]),
        Node(package='nearest_obstacle_shoreline_detector', executable='shoreline_node',
             name='nearest_obstacle_shoreline_detector', output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
