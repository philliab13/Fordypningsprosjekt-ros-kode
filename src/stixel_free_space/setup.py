from glob import glob
from setuptools import setup

setup(
    name='stixel_free_space', version='0.1.0', packages=['stixel_free_space'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/stixel_free_space']),
        ('share/stixel_free_space', ['package.xml', 'README.md', 'requirements-runtime.txt']),
        ('share/stixel_free_space/config', glob('config/*')),
        ('share/stixel_free_space/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='phillip-andre', maintainer_email='philliab@stud.ntnu.no',
    description='ROS 2 streaming adapter and benchmarks for the existing stixel pipeline',
    license='Apache-2.0',
    entry_points={'console_scripts': ['stixel_node = stixel_free_space.node:main','ma2_svo_player = stixel_free_space.ma2_svo_player:main', ]},
)
