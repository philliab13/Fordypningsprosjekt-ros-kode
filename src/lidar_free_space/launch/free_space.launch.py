from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('lidar_free_space'), 'config', 'free_space.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=config),
        DeclareLaunchArgument('target_frame', default_value='gymkhana_task'),
        DeclareLaunchArgument('world_name', default_value='gymkhana_task'),
        DeclareLaunchArgument('publish_world_tf', default_value='true'),
        Node(package='lidar_free_space', executable='world_tf', output='screen',
             condition=IfCondition(LaunchConfiguration('publish_world_tf')),
             parameters=[{'use_sim_time': True,
                          'world_name': LaunchConfiguration('world_name'),
                          'target_frame': LaunchConfiguration('target_frame')}]),
        Node(package='lidar_free_space', executable='estimate_free_space',
             name='lidar_free_space', output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'target_frame': LaunchConfiguration('target_frame')}]),
    ])
