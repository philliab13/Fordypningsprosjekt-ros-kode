import os
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
        Node(package='nearest_obstacle_shoreline_detector', executable='shoreline_node',
             name='nearest_obstacle_shoreline_detector', output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
