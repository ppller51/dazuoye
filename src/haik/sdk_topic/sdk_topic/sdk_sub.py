#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
viewer_node.py  （sdk_sub）
- 订阅 /image_raw，显示/打印图像信息
- 每秒通过参数服务从相机节点读取：曝光(Exposure)、增益(Gain)、帧率(FPS 参数)、像素格式(FOURCC)
- 兼容 ROS 2 Humble：ParameterValue 用 .type 字段（带回退）
- 默认关闭窗口以避免 Wayland/Qt 插件问题；需要时可设 show_window=true 或导出 QT_QPA_PLATFORM=xcb
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.srv import GetParameters, ListParameters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import time


class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')

        # ---- 可调参数（可运行中 ros2 param set 修改）----
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('camera_node', 'opencv_camera')  # 相机发布节点名称（默认配合我给的 camera_node）
        self.declare_parameter('show_window', False)            # Wayland 下建议默认 false，需窗口再改 true

        # 读取参数
        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.camera_node = gp('camera_node').value
        self.show_window = bool(gp('show_window').value)

        # 订阅图像
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self._image_cb, qos)

        # 相机参数服务客户端
        self.cli_list = self.create_client(ListParameters, f'/{self.camera_node}/list_parameters')
        self.cli_get  = self.create_client(GetParameters,  f'/{self.camera_node}/get_parameters')

        # 每秒查询一次相机参数
        self._cam = {'exp': None, 'gain': None, 'drv_fps': None, 'fourcc': None}
        self.create_timer(1.0, self._query_camera_params)

        # 统计接收 FPS
        self._last_t = time.time()
        self._cnt = 0
        self._rx_fps = 0.0

        self.get_logger().info(f"订阅: {self.image_topic} | 相机节点: {self.camera_node} | show_window={self.show_window}")

    # ---------- 工具：兼容式 ParameterValue -> Python 值 ----------
    @staticmethod
    def _pv_to_py(val):
        """
        兼容不同版本的字段名：Humble 用 .type，部分版本可能是 .type_
        类型编码：1=bool, 2=int, 3=double, 4=string
        """
        t = getattr(val, 'type', getattr(val, 'type_', 0))
        if t == 2:
            return int(val.integer_value)
        elif t == 3:
            return float(val.double_value)
        elif t == 4:
            return str(val.string_value)
        elif t == 1:
            return bool(val.bool_value)
        return None

    def _service_ready(self, cli) -> bool:
        if not cli.service_is_ready():
            cli.wait_for_service(timeout_sec=0.0)  # 非阻塞探测
        return cli.service_is_ready()

    # ---------- 轮询相机参数（每秒） ----------
    def _query_camera_params(self):
        if not (self._service_ready(self.cli_list) and self._service_ready(self.cli_get)):
            # 服务未就绪（相机节点可能还没起来）
            return

        # 1) 列出参数名
        req_list = ListParameters.Request()
        req_list.depth = 0
        fut = self.cli_list.call_async(req_list)

        def _after_list(f):
            if f.cancelled() or f.exception() is not None:
                return
            names = set(f.result().result.names)

            # 和 camera_node 保持一致的参数名
            keys = []
            exp = 'exposure' if 'exposure' in names else None
            gn  = 'gain' if 'gain' in names else None
            fps = 'fps' if 'fps' in names else None
            fmt = 'fourcc' if 'fourcc' in names else None

            for k in (exp, gn, fps, fmt):
                if k:
                    keys.append(k)

            if not keys:
                # 相机节点没有这些参数（或者名称不同）
                return

            # 2) 读取参数值
            req_get = GetParameters.Request()
            req_get.names = keys
            fut2 = self.cli_get.call_async(req_get)

            def _after_get(gf):
                if gf.cancelled() or gf.exception() is not None:
                    return
                vals = gf.result().values
                parsed = {k: self._pv_to_py(v) for k, v in zip(keys, vals)}
                self._cam['exp']    = parsed.get('exposure')
                self._cam['gain']   = parsed.get('gain')
                self._cam['drv_fps']= parsed.get('fps')
                self._cam['fourcc'] = parsed.get('fourcc')

            fut2.add_done_callback(_after_get)

        fut.add_done_callback(_after_list)

    # ---------- 图像回调 ----------
    def _image_cb(self, msg: Image):
        # 统计接收 FPS（每秒刷新一次）
        self._cnt += 1
        now = time.time()
        if now - self._last_t >= 1.0:
            self._rx_fps = self._cnt / (now - self._last_t)
            self._cnt = 0
            self._last_t = now
            # 终端打印
            print(
                f"[INFO] 分辨率:{msg.width}x{msg.height} | 实时FPS(接收):{self._rx_fps:.2f} | ROS编码:{msg.encoding} | "
                f"曝光:{self._cam['exp']} | 增益:{self._cam['gain']} | 帧率(驱动参数):{self._cam['drv_fps']} | "
                f"像素格式(驱动参数):{self._cam['fourcc']}"
            )

        # 可选：窗口显示（Wayland 下如有 Qt 插件问题，保持 false）
        if not self.show_window:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败：{e}')
            return

        text1 = f"{msg.width}x{msg.height} | RX:{self._rx_fps:.1f} fps | enc:{msg.encoding}"
        text2 = f"Exp:{self._cam['exp']} Gain:{self._cam['gain']} FPS(drv):{self._cam['drv_fps']} FOURCC:{self._cam['fourcc']}"
        cv2.putText(frame, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
        cv2.imshow('viewer', frame); cv2.waitKey(1)

    def destroy_node(self):
        # 关闭窗口（即使没开也安全）
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
