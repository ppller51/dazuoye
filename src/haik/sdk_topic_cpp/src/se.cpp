#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <future>  // 包含 std::async

using namespace std::chrono_literals;

#ifdef __cplusplus
extern "C" {
#endif
#include "MvCameraControl.h"
#ifdef __cplusplus
}
#endif

// ========== 工具：从设备信息取序列号 / IP ==========
static std::string GetSerial(const MV_CC_DEVICE_INFO* p) {
  if (!p) return {};
  if (p->nTLayerType == MV_GIGE_DEVICE) {
    return std::string(reinterpret_cast<const char*>(p->SpecialInfo.stGigEInfo.chSerialNumber));
  } else if (p->nTLayerType == MV_USB_DEVICE) {
    return std::string(reinterpret_cast<const char*>(p->SpecialInfo.stUsb3VInfo.chSerialNumber));
  }
  return {};
}

static std::string GetGigEIp(const MV_CC_DEVICE_INFO* p) {
  if (!p || p->nTLayerType != MV_GIGE_DEVICE) return {};
  const auto &g = p->SpecialInfo.stGigEInfo;
  char ip[32]{0};
  std::snprintf(ip, sizeof(ip), "%u.%u.%u.%u",
    (g.nCurrentIp & 0xff),
    (g.nCurrentIp >> 8) & 0xff,
    (g.nCurrentIp >> 16) & 0xff,
    (g.nCurrentIp >> 24) & 0xff);
  return std::string(ip);
}

// ========== 像素格式映射 ==========
// - camera_enum: 提给 MV_CC_SetEnumValue("PixelFormat", camera_enum)
// - dst_pixel_type: MV_CC_ConvertPixelType 目标类型
// - publish_encoding: ROS Image.encoding
struct PixelFmtMap {
  unsigned int    camera_enum = 0;                            // MV_CC_SetEnumValue 用
  MvGvspPixelType dst_pixel_type = PixelType_Gvsp_BGR8_Packed; // 转换目标
  const char*     publish_encoding = "bgr8";
};

// 把用户参数字符串映射到具体枚举
static bool MapPixelFormat(const std::string& key, PixelFmtMap& out) {
  const std::string k = key; // 直接大小写敏感；需要可加 tolower
  if (k == "BGR8") {
    out.camera_enum     = static_cast<unsigned int>(PixelType_Gvsp_BGR8_Packed);
    out.dst_pixel_type  = PixelType_Gvsp_BGR8_Packed;
    out.publish_encoding= "bgr8";
    return true;
  }
  if (k == "RGB8") {
    out.camera_enum     = static_cast<unsigned int>(PixelType_Gvsp_RGB8_Packed);
    out.dst_pixel_type  = PixelType_Gvsp_RGB8_Packed;
    out.publish_encoding= "rgb8";
    return true;
  }
  if (k == "Mono8") {
    out.camera_enum     = static_cast<unsigned int>(PixelType_Gvsp_Mono8);
    out.dst_pixel_type  = PixelType_Gvsp_Mono8;
    out.publish_encoding= "mono8";
    return true;
  }
  return false;
}

class HikUsbBgr8Pub : public rclcpp::Node {
public:
  HikUsbBgr8Pub()
  : Node("hik_usb_bgr8_pub")
  {
    // 连接/发布相关参数
    serial_         = declare_parameter<std::string>("serial_number", "00E89503815");
    ip_             = declare_parameter<std::string>("ip_address", "");
    frame_id_       = declare_parameter<std::string>("frame_id", "camera_frame");
    image_topic_    = declare_parameter<std::string>("image_topic", "image_raw");
    qos_reliable_   = declare_parameter<bool>("qos_reliable", false); // false=BestEffort(推荐)，true=Reliable
    reconnect_ms_   = declare_parameter<int>("reconnect_interval_ms", 1000);

    // 相机控制参数（自动优先，手动在 auto=Off 时生效）
    exposure_auto_  = declare_parameter<int>("exposure_auto", 0);      // 0=Off,1=Once,2=Continuous
    exposure_time_  = declare_parameter<double>("exposure_time", 5000.0); // 微秒
    gain_auto_      = declare_parameter<int>("gain_auto", 0);          // 0=Off,1=Once,2=Continuous
    gain_           = declare_parameter<double>("gain", 6.0);
    acq_fps_        = declare_parameter<double>("acquisition_fps", 180.0);
    pixel_format_   = declare_parameter<std::string>("pixel_format", "BGR8");

    // 依据 pixel_format_ 初始化映射
    {
      PixelFmtMap m{};
      if (!MapPixelFormat(pixel_format_, m)) {
        RCLCPP_WARN(get_logger(), "Unknown pixel_format '%s', fallback to BGR8", pixel_format_.c_str());
        pixel_format_ = "BGR8";
        MapPixelFormat(pixel_format_, m);
      }
      camera_pixel_enum_ = m.camera_enum;
      dst_pixel_type_    = m.dst_pixel_type;
      publish_encoding_  = m.publish_encoding;
    }

    // 发布者（QoS 可切换）
    {
      rclcpp::QoS qos(rclcpp::KeepLast(10));
      if (qos_reliable_) qos.reliable(); else qos.best_effort();
      pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic_, qos);
    }

    // 参数动态回调
    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HikUsbBgr8Pub::onParamSet, this, std::placeholders::_1));

    MV_CC_Initialize();          // SDK 初始化
    ensureConnected();           // 立即尝试一次

    // 创建一个定时器，每1秒调用一次输出帧率的回调
    frame_rate_timer_ = create_wall_timer(
      std::chrono::seconds(1), std::bind(&HikUsbBgr8Pub::printFrameRate, this));

    timer_ = create_wall_timer(  // 周期自检/重连
      std::chrono::milliseconds(reconnect_ms_),
      std::bind(&HikUsbBgr8Pub::ensureConnected, this));
  }

  ~HikUsbBgr8Pub() override {
    stopGrab();
    closeDevice();
    MV_CC_Finalize();
  }

private:
  // ========== 连接 / 重连 ==========
  void ensureConnected() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (connected_) return;

    closeDeviceNoLock();
    if (openBySnOrIp(serial_, ip_)) {
      connected_ = true;
      startGrab();
      RCLCPP_INFO(get_logger(), "Connected to camera (SN=%s, IP=%s) and started grabbing.",
                  serial_.c_str(), ip_.c_str());
    }
  }

  bool openBySnOrIp(const std::string& serial, const std::string& ip) {
    MV_CC_DEVICE_INFO_LIST list{};
    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
    if (ret != MV_OK || list.nDeviceNum == 0) {
      RCLCPP_WARN(get_logger(), "EnumDevices failed or no camera found (ret=0x%x).", ret);
      return false;
    }

    int idx = -1;
    for (unsigned int i = 0; i < list.nDeviceNum; ++i) {
      auto* info = list.pDeviceInfo[i];
      const auto sn = GetSerial(info);
      const auto ipstr = GetGigEIp(info);
      if (!serial.empty() && sn == serial) { idx = static_cast<int>(i); break; }
      if (idx < 0 && !ip.empty() && ipstr == ip) { idx = static_cast<int>(i); }
    }
    if (idx < 0) idx = 0; // 都没指定：取第一台

    // 句柄 + 打开
    ret = MV_CC_CreateHandle(&handle_, list.pDeviceInfo[idx]);
    if (ret != MV_OK || !handle_) {
      RCLCPP_ERROR(get_logger(), "CreateHandle failed: 0x%x", ret);
      handle_ = nullptr;
      return false;
    }
    ret = MV_CC_OpenDevice(handle_);
    if (ret != MV_OK) {
      RCLCPP_ERROR(get_logger(), "OpenDevice failed: 0x%x", ret);
      MV_CC_DestroyHandle(handle_); handle_ = nullptr; return false;
    }

    // 异常回调（断线）
    MV_CC_RegisterExceptionCallBack(handle_, &HikUsbBgr8Pub::OnException, this);

    // 采集模式 + 帧率（若某些型号返回非 MV_OK，可忽略继续）
    MV_CC_SetEnumValue(handle_, "AcquisitionMode", 2); // 2=Continuous
    MV_CC_SetEnumValue(handle_, "TriggerMode", 0);     // 0=Off
    MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", acq_fps_);

    // 曝光/增益（先设自动，再在手动模式下设值）
    MV_CC_SetEnumValue(handle_, "ExposureAuto", exposure_auto_);
    if (exposure_auto_ == 0) MV_CC_SetFloatValue(handle_, "ExposureTime", exposure_time_);
    MV_CC_SetEnumValue(handle_, "GainAuto", gain_auto_);
    if (gain_auto_ == 0) MV_CC_SetFloatValue(handle_, "Gain", gain_);

    // 相机端像素格式（按参数设置）
    if (camera_pixel_enum_ != 0) {
      int ret_pf = MV_CC_SetEnumValue(handle_, "PixelFormat", camera_pixel_enum_);
      if (ret_pf != MV_OK) {
        RCLCPP_WARN(get_logger(), "Set PixelFormat(0x%x) failed: 0x%x, keep device default.",
                    camera_pixel_enum_, ret_pf);
      }
    }

    ret = MV_CC_StartGrabbing(handle_);
    if (ret != MV_OK) {
      RCLCPP_ERROR(get_logger(), "StartGrabbing failed: 0x%x", ret);
      MV_CC_CloseDevice(handle_);
      MV_CC_DestroyHandle(handle_);
      handle_ = nullptr;
      return false;
    }
    return true;
  }

  void closeDevice() {
    std::lock_guard<std::mutex> lk(mtx_);
    closeDeviceNoLock();
  }
  void closeDeviceNoLock() {
    if (handle_) {
      MV_CC_StopGrabbing(handle_);
      MV_CC_CloseDevice(handle_);
      MV_CC_DestroyHandle(handle_);
      handle_ = nullptr;
    }
    connected_ = false;
  }

  static void __stdcall OnException(unsigned int msgType, void* pUser) {
    auto* self = static_cast<HikUsbBgr8Pub*>(pUser);
    if (!self) return;
    RCLCPP_ERROR(self->get_logger(), "Camera exception: 0x%x (will reconnect)", msgType);
    self->connected_ = false;
    self->stopGrab();
    self->closeDevice();
  }

  // 新增定时器回调方法，输出当前帧率
  void printFrameRate() {
    if (!handle_) return;

    MVCC_FLOATVALUE frameRate{};
    int ret = MV_CC_GetFrameRate(handle_, &frameRate);
    if (ret == MV_OK) {
      RCLCPP_INFO(get_logger(), "Current frame rate: %.2f fps", frameRate.fCurValue);
    } else {
      RCLCPP_WARN(get_logger(), "Get frame rate failed! Error code: 0x%x", ret);
    }
  }

  // ========== 抓帧并发布 ==========
  void startGrab() {
    stopGrab();
    running_.store(true);
    th_ = std::thread([this]{ this->grabLoop(); });
  }
  void stopGrab() {
    running_.store(false);
    if (th_.joinable()) th_.join();
  }

  void grabLoop() {
    std::vector<uint8_t> out_buf;  // 转换缓冲
    while (rclcpp::ok() && running_.load()) {
      if (!connected_ || !handle_) { 
          std::this_thread::sleep_for(10ms); 
          continue; 
      }

      MV_FRAME_OUT frame{}; // 清零
      int ret = MV_CC_GetImageBuffer(handle_, &frame, 1000); // 1s 超时
      if (ret != MV_OK) continue;

      // 异步执行图像处理，并获取 future 返回值
        std::future<void> result = std::async(std::launch::async, &HikUsbBgr8Pub::processFrame, this, frame);

        // 如果需要，可以等待异步任务完成
        result.get();  // 等待异步任务完成，阻塞直到完成

      MV_CC_FreeImageBuffer(handle_, &frame);
    }
  }

  void processFrame(const MV_FRAME_OUT& frame) {
    const auto& fi = frame.stFrameInfo;
    MvGvspPixelType dst_pix;
    const char* enc;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      dst_pix = dst_pixel_type_;
      enc     = publish_encoding_;
    }

    // 根据目标编码计算 step
    size_t pixel_size = (enc && std::string(enc) == "mono8") ? 1 : 3;
    size_t need = static_cast<size_t>(fi.nWidth) * fi.nHeight * pixel_size;
        std::vector<uint8_t> out_buf(need);

    MV_CC_PIXEL_CONVERT_PARAM conv{};
    conv.nWidth         = fi.nWidth;
    conv.nHeight        = fi.nHeight;
    conv.pSrcData       = frame.pBufAddr;
    conv.nSrcDataLen    = fi.nFrameLen;
    conv.enSrcPixelType = fi.enPixelType;
    conv.enDstPixelType = dst_pix;
    conv.pDstBuffer     = out_buf.data();
    conv.nDstBufferSize = static_cast<unsigned int>(out_buf.size());

    int ret = MV_CC_ConvertPixelType(handle_, &conv);
    if (ret != MV_OK) {
      // 如果直接就是期望格式，可尝试直拷（例如设备原生 BGR8/Mono8）
      if (fi.enPixelType == dst_pix && fi.nFrameLen >= need) {
        std::memcpy(out_buf.data(), frame.pBufAddr, need);
      } else {
        RCLCPP_WARN(get_logger(), "ConvertPixelType failed: 0x%x", ret);
        return;
      }
    }

    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp = now();
    msg.header.frame_id = frame_id_;
    msg.height = fi.nHeight;
    msg.width  = fi.nWidth;
    msg.encoding = enc ? enc : "bgr8";
    msg.is_bigendian = false;
    msg.step = fi.nWidth * pixel_size;
    msg.data.assign(out_buf.begin(), out_buf.end());

    pub_->publish(std::move(msg));
  }

  // ========== 动态参数回调 ==========
  rcl_interfaces::msg::SetParametersResult
  onParamSet(const std::vector<rclcpp::Parameter>& params) {
    std::lock_guard<std::mutex> lk(mtx_);
    rcl_interfaces::msg::SetParametersResult res; res.successful = true; res.reason = "ok";

    for (const auto& p : params) {
      try {
        if (p.get_name() == "exposure_auto") {
          exposure_auto_ = p.as_int();
          if (handle_) {
            MV_CC_SetEnumValue(handle_, "ExposureAuto", exposure_auto_);
            if (exposure_auto_ == 0) MV_CC_SetFloatValue(handle_, "ExposureTime", exposure_time_);
          }
        } else if (p.get_name() == "exposure_time") {
          exposure_time_ = p.as_double();
          if (handle_ && exposure_auto_ == 0) {
            MV_CC_SetFloatValue(handle_, "ExposureTime", exposure_time_);
          }
        } else if (p.get_name() == "gain_auto") {
          gain_auto_ = p.as_int();
          if (handle_) {
            MV_CC_SetEnumValue(handle_, "GainAuto", gain_auto_);
            if (gain_auto_ == 0) MV_CC_SetFloatValue(handle_, "Gain", gain_);
          }
        } else if (p.get_name() == "gain") {
          gain_ = p.as_double();
          if (handle_ && gain_auto_ == 0) {
            MV_CC_SetFloatValue(handle_, "Gain", gain_);
          }
        } else if (p.get_name() == "acquisition_fps") {
          acq_fps_ = p.as_double();
          if (handle_) {
            MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
            MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", acq_fps_);
          }
        } else if (p.get_name() == "pixel_format") {
          const std::string req = p.as_string();
          PixelFmtMap m{};
          if (!MapPixelFormat(req, m)) {
            res.successful = false; res.reason = "unsupported pixel_format";
            continue;
          }
          // 应用到设备：需要停采并重启
          if (handle_) {
            MV_CC_StopGrabbing(handle_);
            int r = MV_CC_SetEnumValue(handle_, "PixelFormat", m.camera_enum);
            if (r != MV_OK) {
              // 恢复采集，返回失败
              MV_CC_StartGrabbing(handle_);
              res.successful = false; res.reason = "Set PixelFormat failed";
              continue;
            }
            int r2 = MV_CC_StartGrabbing(handle_);
            if (r2 != MV_OK) {
              res.successful = false; res.reason = "Restart grabbing failed";
              continue;
            }
          }
          // 本地生效
          pixel_format_      = req;
          camera_pixel_enum_ = m.camera_enum;
          dst_pixel_type_    = m.dst_pixel_type;
          publish_encoding_  = m.publish_encoding;
          RCLCPP_INFO(get_logger(), "PixelFormat set to %s (camera=0x%x, publish=%s)",
                      pixel_format_.c_str(), camera_pixel_enum_, publish_encoding_);
        } else if (p.get_name() == "qos_reliable" || p.get_name() == "image_topic") {
          // 运行时变更发布 QoS/话题名需要重建 publisher，通常留到下次重启
          res.successful = false; res.reason = "Changing QoS/topic at runtime is not supported";
        }
      } catch (...) {
        res.successful = false; res.reason = "parameter set error";
      }
    }
    return res;
  }

private:
  // 连接/发布参数
  std::string serial_;
  std::string ip_;
  std::string frame_id_;
  std::string image_topic_;
  bool        qos_reliable_{false};
  int         reconnect_ms_{2000};

  // 相机控制参数
  int     exposure_auto_{2};
  double  exposure_time_{8000.0};
  int     gain_auto_{2};
  double  gain_{6.0};
  double  acq_fps_{30.0};
  std::string pixel_format_{"BGR8"};

  // 像素格式运行时设置
  unsigned int    camera_pixel_enum_{static_cast<unsigned int>(PixelType_Gvsp_BGR8_Packed)};
  MvGvspPixelType dst_pixel_type_{PixelType_Gvsp_BGR8_Packed};
  const char*     publish_encoding_{"bgr8"};

  // 设备状态
  std::mutex mtx_;
  void* handle_ = nullptr;
  std::atomic<bool> connected_{false};

  // 抓取线程
  std::atomic<bool> running_{false};
  std::thread th_;

  // ROS
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr frame_rate_timer_;  // 新增的定时器用于打印帧率
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HikUsbBgr8Pub>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

