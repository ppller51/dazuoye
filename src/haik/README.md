colcon build

ros2 launch sdk_launch camera.launch.py

//改曝光：  
ros2 param set /hik_usb_bgr8_pub exposure_auto 0
ros2 param set /hik_usb_bgr8_pub exposure_time 8000.0
    
//改增益：  
ros2 param set /hik_usb_bgr8_pub gain_auto 0 
ros2 param set /hik_usb_bgr8_pub gain 6.0
    
//改帧率：  
ros2 param set /hik_usb_bgr8_pub acquisition_fps 60.0
    
//改相机端像素格式（会停采/重启）：  
ros2 param set /hik_usb_bgr8_pub pixel_format BGR8
ros2 param set /hik_usb_bgr8_pub pixel_format RGB8
ros2 param set /hik_usb_bgr8_pub pixel_format Mono8



神经训练见“shuzi_juanji:model.py”

sdk_topic_cpp/se.cpp __________获取相机图像
sdk_topic_cpp/gr.cpp___________处理图像
sdk_topic/bd.py_________________相机标定
