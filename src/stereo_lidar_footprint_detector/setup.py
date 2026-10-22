from setuptools import setup

setup(name='stereo_lidar_footprint_detector', version='0.1.0',
      packages=['stereo_lidar_footprint_detector'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/stereo_lidar_footprint_detector']),
                  ('share/stereo_lidar_footprint_detector', ['package.xml', 'README.md'])],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='phillip-andre', maintainer_email='philliab@stud.ntnu.no',
      description='Square obstacle polygons from registered camera depth and LiDAR',
      license='Apache-2.0',
      entry_points={'console_scripts': [
          'footprint_node = stereo_lidar_footprint_detector.node:main',
          'test_obstacle_node = stereo_lidar_footprint_detector.test_obstacle_node:main',
          'bev_node = stereo_lidar_footprint_detector.bev_node:main',
          'frame_review_node = stereo_lidar_footprint_detector.frame_review_node:main']})
