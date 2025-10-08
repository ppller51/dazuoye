#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class FPSPrinter(Node):
    def __init__(self):
        super().__init__('fps_printer')

        # 订阅话题参数
        self.declare_parameter('topic', '/image_raw')
        topic = self.get_parameter('topic').value

        # QoS：相机发布端通常是 Best Effort
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Image, topic, self.callback, qos)
        self.times = deque(maxlen=60)
        self.last_print = time.time()

        self.get_logger().info(f"Subscribed to {topic} (for FPS display)")

    def callback(self, msg):
        now = time.time()
        self.times.append(now)
        if len(self.times) < 2:
            return
        dt = (self.times[-1] - self.times[0]) / (len(self.times) - 1)
        fps = 1.0 / dt if dt > 0 else 0.0

        # 每2秒打印一次，防止太频繁
        if now - self.last_print > 2.0:
            self.get_logger().info(f"Receiving {fps:.1f} FPS | {msg.width}x{msg.height} {msg.encoding}")
            self.last_print = now


def main():
    rclpy.init()
    node = FPSPrinter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



