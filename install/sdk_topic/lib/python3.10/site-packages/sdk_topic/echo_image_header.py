#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import time

class ImageEcho(Node):
    def __init__(self):
        super().__init__('image_echo_best_effort')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(Image, '/image_raw', self.cb, qos)
        self.fps_pub = self.create_publisher(Float32, 'fps', 10)
        
        self.frame_count = 0
        self.last_time = self.get_clock().now()

    def cb(self, msg: Image):
        self.frame_count += 1
        now = self.get_clock().now()
        elapsed = (now - self.last_time).nanoseconds / 1e9  # 转换为秒
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            fps_msg = Float32()
            fps_msg.data = fps
            self.fps_pub.publish(fps_msg)
            self.get_logger().info(f'实际帧率: {fps:.2f} FPS')
            
            self.frame_count = 0
            self.last_time = now

def main():
    rclpy.init()
    node = ImageEcho()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
