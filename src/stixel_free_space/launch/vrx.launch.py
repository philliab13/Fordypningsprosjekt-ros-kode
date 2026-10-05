"""Attach to an existing VRX simulation; no second simulator is started."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('stixel_free_space'), 'config', 'vrx.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=config),
        DeclareLaunchArgument('repo_path', default_value=str(
            Path.home() / 'Documents/Stixel_based_free_space_estimation/free-space-estimation')),
        DeclareLaunchArgument('python_executable', default_value='/usr/bin/python3',
                              description='ROS-compatible Python, optionally from a --system-site-packages venv'),
        DeclareLaunchArgument('world_name', default_value='gymkhana_task'),
        DeclareLaunchArgument('fixed_frame', default_value=LaunchConfiguration('world_name')),
        DeclareLaunchArgument('publish_world_tf', default_value='false',
                              description='Enable only if lidar_free_space/world_tf is not already running'),
        DeclareLaunchArgument('retina_masks', default_value='true'),
        DeclareLaunchArgument('publish_images', default_value='true'),
        DeclareLaunchArgument('publish_markers', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        Node(package='lidar_free_space', executable='world_tf', output='screen',
             condition=IfCondition(LaunchConfiguration('publish_world_tf')),
             parameters=[{'use_sim_time': True,
                          'world_name': LaunchConfiguration('world_name'),
                          'target_frame': LaunchConfiguration('fixed_frame')}]),
        Node(package='stixel_free_space', executable='stixel_node', name='stixel_free_space',
             output='screen', prefix=[LaunchConfiguration('python_executable')],
             parameters=[LaunchConfiguration('params_file'), {
                 'repo_path': LaunchConfiguration('repo_path'),
                 'fixed_frame': LaunchConfiguration('fixed_frame'),
                 'retina_masks': ParameterValue(LaunchConfiguration('retina_masks'), value_type=bool),
                 'publish_images': ParameterValue(LaunchConfiguration('publish_images'), value_type=bool),
                 'publish_markers': ParameterValue(LaunchConfiguration('publish_markers'), value_type=bool),
             }]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(get_package_share_directory('stixel_free_space'),
                                          'config', 'stixels.rviz')],
             parameters=[{'use_sim_time': True}]),
    ])
