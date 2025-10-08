#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class ImagePublisher(Node):
    def __init__(self, name):
        super().__init__(name)

        # 可选：设备号/帧率做成参数（默认0号摄像头、10Hz）
        self.declare_parameter('device_index', 0)
        self.declare_parameter('fps', 10.0)
        self.device_index = int(self.get_parameter('device_index').value)
        self.fps = float(self.get_parameter('fps').value)
        period = 0.0 if self.fps <= 0 else 1.0 / self.fps

        # 传感器流QoS（与RViz设置Best Effort匹配更稳）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(Image, 'image_raw', qos)

        # 打开摄像头（Linux下可显式V4L2；跨平台留默认也行）
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        except Exception:
            self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {self.device_index}')
        self.bridge = CvBridge()

        self.timer = self.create_timer(period, self.timer_callback)

        # 打印实际发布速率（防刷屏，2秒一次）
        self._last = time.time()
        self._cnt = 0

    def timer_callback(self):
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('读取帧失败')
            return

        # 转ROS消息，补时间戳与frame_id，编码用bgr8
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher_.publish(msg)

        # 简单速率日志（2秒一次）
        self._cnt += 1
        now = time.time()
        if now - self._last > 2.0:
            fps_run = self._cnt / (now - self._last)
            self.get_logger().info(f'Publishing /image_raw @ {fps_run:.1f} FPS')
            self._cnt = 0
            self._last = now

    def destroy_node(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher("topic_webcam_pub")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
