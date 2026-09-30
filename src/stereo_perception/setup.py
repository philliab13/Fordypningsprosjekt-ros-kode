from setuptools import find_packages, setup

package_name = 'stereo_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phillip-andre',
    maintainer_email='philliab@stud.ntnu.no',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bayesod_camera_bridge = stereo_perception.bayesod_camera_bridge:main',
            'stereo_viewer = stereo_perception.stereo_viewer:main',
            'wamv_teleop = stereo_perception.wamv_teleop:main',
        ],
    },
)
