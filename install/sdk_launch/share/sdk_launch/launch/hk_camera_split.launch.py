from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch_ros.actions import Node
import os

def generate_launch_description():
    cap_yaml = LaunchConfiguration('capture_yaml')
    view_yaml = LaunchConfiguration('viewer_yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'capture_yaml',
            default_value=os.path.join(
                os.getenv('COLCON_CURRENT_PREFIX', ''),  # 运行时可为空
                'share', 'hk_camera_ros2', 'config', 'capture.yaml'
            )
        ),
        DeclareLaunchArgument(
            'viewer_yaml',
            default_value=os.path.join(
                os.getenv('COLCON_CURRENT_PREFIX', ''),
                'share', 'hk_camera_ros2', 'config', 'viewer.yaml'
            )
        ),

        Node(
            package='hk_camera_ros2',
            executable='hk_camera_capture_node',
            name='hk_camera_capture',
            output='screen',
            parameters=[cap_yaml]
        ),

        Node(
            package='hk_camera_ros2',
            executable='hk_image_viewer_node',
            name='hk_image_viewer',
            output='screen',
            parameters=[view_yaml]
        )
    ])
