from glob import glob
from setuptools import setup

setup(
    name='nearest_obstacle_shoreline_detector', version='0.1.0',
    packages=['nearest_obstacle_shoreline_detector'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/nearest_obstacle_shoreline_detector']),
        ('share/nearest_obstacle_shoreline_detector', ['package.xml', 'readme.md', 'requirements-runtime.txt']),
        ('share/nearest_obstacle_shoreline_detector/config', glob('config/*.yaml')),
        ('share/nearest_obstacle_shoreline_detector/launch', glob('launch/*.launch.py'))],
    install_requires=['setuptools', 'numpy', 'scipy'], zip_safe=True,
    maintainer='phillip-andre', maintainer_email='philliab@stud.ntnu.no',
    description='LiDAR-ranged shoreline barriers from fused water segmentation',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'shoreline_node = nearest_obstacle_shoreline_detector.node:main',
        'lidar_projection_node = nearest_obstacle_shoreline_detector.lidar_projection_node:main']},
)
