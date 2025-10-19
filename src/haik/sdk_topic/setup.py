from setuptools import find_packages, setup

package_name = 'sdk_topic'

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
    maintainer='pl',
    maintainer_email='13367748435@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bd =sdk_topic.bd:main',
            'cameracalibrator = camera_calibration.nodes.cameracalibrator:main',
            'cameracheck = camera_calibration.nodes.cameracheck:main',
        ],
    },
)
