from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='ma2_zed_static_tf',
            arguments=[
                '--x', '-4.02190564',
                '--y', '-0.89386966',
                '--z', '-1.09727816',
                '--qx', '-0.47626201',
                '--qy', '0.46569084',
                '--qz', '0.53470923',
                '--qw', '-0.51999286',
                '--frame-id', 'p0_floor',
                '--child-frame-id', 'ma2_zed_left_optical',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='ma2_water_plane_static_tf',
            arguments=[
                '--x', '0',
                '--y', '0',
                # Floor z points down. Camera z=-1.09727816 m and
                # assumed vertical camera-to-water distance is 1.6 m.
                '--z', '0.50272184',
                '--roll', '0',
                '--pitch', '0',
                '--yaw', '0',
                '--frame-id', 'p0_floor',
                '--child-frame-id', 'ma2_water_plane',
            ],
        ),
    ])
