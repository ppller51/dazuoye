# /home/pl/ros2_ws1/src/haik/sdk_launch/launch/camera.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---- Launch 参数（可在命令行覆盖） ----
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/home/pl/ros2_ws1/src/haik/sdk_launch/config/camera.yaml',
        description='YAML 参数文件，提供相机连接与采集参数'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='/home/pl/ros2_ws1/src/haik/sdk_topic_cpp/sdk_topic_cpp.rviz',
        description='RViz2 配置文件路径'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='是否同时启动 RViz2（true/false）'
    )

    # 新增：是否启用FPS打印节点、以及订阅的话题名
    fps_arg = DeclareLaunchArgument(
        'fps', default_value='true',
        description='是否启动终端FPS打印节点（sdk_topic/print_fps）'
    )
    fps_topic_arg = DeclareLaunchArgument(
        'fps_topic', default_value='/image_raw',
        description='FPS打印节点订阅的话题名'
    )

    # ---- 相机发布节点（读取参数文件） ----
    camera_node = Node(
        package='sdk_topic_cpp',
        executable='se',
        name='hik_usb_bgr8_pub',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        # remappings=[('image_raw', '/camera/image_raw')],  # 如需改名可放开
    )


    # ---- RViz2（按需启动） ----
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        params_file_arg,
        rviz_config_arg,
        use_rviz_arg,
        camera_node,
        rviz_node
    ])
