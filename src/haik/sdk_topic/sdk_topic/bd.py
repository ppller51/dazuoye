#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os

class CameraCalibrator(Node):
    def __init__(self):
        # ROS 2节点初始化
        super().__init__('camera_calibrator')
        
        # 标定板参数 (28.9mm为每个格子的边长)
        self.board_size = (9, 6)  # 内角点数量（根据实际标定板修改）
        self.square_size = 0.0289  # 单位：米
        
        # 存储标定数据
        self.obj_points = []  # 3D世界坐标点
        self.img_points = []  # 2D图像坐标点
        
        # 生成标定板的3D坐标
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 图像订阅器（ROS 2的订阅方式）
        self.image_sub = self.create_subscription(
            Image, 
            "/camera/image_raw",  # 图像话题（根据实际话题修改）
            self.image_callback, 
            10  # QoS配置
        )
        
        # 标定结果保存路径
        self.calibration_path = self.declare_parameter("calibration_path", "./calibration_data").value
        if not os.path.exists(self.calibration_path):
            os.makedirs(self.calibration_path)
        
        # 已收集的标定板图像数量
        self.sample_count = 0
        self.required_samples = self.declare_parameter("required_samples", 20).value  # 建议至少20张

        # 声明图像话题参数（默认值改为 /image_raw）
        self.declare_parameter('image_topic', '/image_raw')  # 这里修改默认话题
        self.image_topic = self.get_parameter('image_topic').value
        
        
        self.get_logger().info("相机标定节点已启动，正在等待图像...")
        self.get_logger().info(f"需要收集 {self.required_samples} 张标定板图像")
        self.get_logger().info(f"标定板规格: {self.board_size[0]}x{self.board_size[1]} 内角点, 格子大小 {self.square_size*1000}mm")

    def image_callback(self, data):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return
        
        # 查找标定板角点
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        # 如果找到角点
        if ret:
            # 亚像素级角点检测，提高精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 存储3D和2D点
            self.obj_points.append(self.objp)
            self.img_points.append(corners2)
            
            # 绘制角点并显示
            cv2.drawChessboardCorners(cv_image, self.board_size, corners2, ret)
            self.sample_count += 1
            self.get_logger().info(f"已收集 {self.sample_count}/{self.required_samples} 张有效图像")
            
            # 当收集到足够的样本时进行标定
            if self.sample_count >= self.required_samples:
                self.calibrate_camera(gray.shape[::-1])
                rclpy.shutdown()  # 标定完成后关闭节点
        
        # 显示图像
        cv2.imshow('Camera Calibration', cv_image)
        cv2.waitKey(1)

    def calibrate_camera(self, image_size):
        """执行相机标定并保存结果"""
        self.get_logger().info("开始相机标定...")
        
        # 执行标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, image_size, None, None
        )
        
        if not ret:
            self.get_logger().error("标定失败，请重新收集图像样本")
            return
        
        # 计算重投影误差（评估标定精度）
        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        
        mean_error /= len(self.obj_points)
        self.get_logger().info(f"重投影平均误差: {mean_error:.6f} 像素 (值越小精度越高)")
        
        # 保存标定结果
        calibration_data = {
            "camera_matrix": mtx,
            "dist_coeff": dist,
            "image_size": image_size,
            "reprojection_error": mean_error
        }
        
        np.savez(
            os.path.join(self.calibration_path, "camera_calibration.npz"),
            **calibration_data
        )
        
        self.get_logger().info("标定完成，结果已保存至: %s", 
                     os.path.join(self.calibration_path, "camera_calibration.npz"))
        self.get_logger().info("相机内参矩阵:")
        self.get_logger().info("\n%s", mtx)
        self.get_logger().info("畸变系数:")
        self.get_logger().info("%s", dist)

def main(args=None):
    # ROS 2初始化
    rclpy.init(args=args)
    # 创建节点实例
    calibrator = CameraCalibrator()
    # 运行节点
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断
        pass
    finally:
        # 关闭窗口和节点
        cv2.destroyAllWindows()
        calibrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
