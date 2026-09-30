from glob import glob
from setuptools import setup
setup(name='lidar_free_space', version='0.1.0', packages=['lidar_free_space'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/lidar_free_space']),
                  ('share/lidar_free_space', ['package.xml', 'README.md']),
                  ('share/lidar_free_space/config', glob('config/*')),
                  ('share/lidar_free_space/launch', glob('launch/*'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='phillip-andre', maintainer_email='philliab@stud.ntnu.no',
      description='Conservative polar lidar boundary visualization for VRX', license='Apache-2.0',
      entry_points={'console_scripts': ['estimate_free_space = lidar_free_space.node:main', 'world_tf = lidar_free_space.world_tf:main']})
