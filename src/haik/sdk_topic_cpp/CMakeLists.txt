cmake_minimum_required(VERSION 3.8)
project(sdk_topic_cpp)

# === 关键：强制使用系统 Python，彻底绕开残留的 anaconda 路径 ===
# 这样即使外部环境有旧缓存，CMake/ament 也会用 /usr/bin/python3
set(AMENT_PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python used by ament" FORCE)
set(PYTHON_EXECUTABLE       "/usr/bin/python3" CACHE FILEPATH "Python used by CMake"  FORCE)
set(Python3_EXECUTABLE      "/usr/bin/python3" CACHE FILEPATH "Python3 interpreter"   FORCE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_srvs REQUIRED)  # 添加这一行


# === 关键：显式找并链接 libcurl（修复 CURL_OPENSSL_4 符号问题）===
find_package(CURL REQUIRED)

# ==== 海康 MVS SDK 路径（可用环境变量覆盖）====
if(NOT DEFINED ENV{HIK_SDK_ROOT})
  set(ENV{HIK_SDK_ROOT} "/opt/MVS")
endif()
set(HIK_SDK_ROOT $ENV{HIK_SDK_ROOT})

# 头文件目录
include_directories(${HIK_SDK_ROOT}/include)

# 指定库的绝对路径
set(MVS_LIB_PATH "${HIK_SDK_ROOT}/lib/64")
set(MVS_LIB "${MVS_LIB_PATH}/libMvCameraControl.so")
if(NOT EXISTS "${MVS_LIB}")
  message(FATAL_ERROR "未找到海康库: ${MVS_LIB}；请确认 HIK_SDK_ROOT=${HIK_SDK_ROOT}")
endif()
message(STATUS "Hik SDK include: ${HIK_SDK_ROOT}/include")
message(STATUS "Hik SDK lib:     ${MVS_LIB}")

# 线程库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ================== 可执行文件 ==================
add_executable(se src/se.cpp)
ament_target_dependencies(se rclcpp sensor_msgs)
target_link_libraries(se
  ${MVS_LIB}
  Threads::Threads
  dl
  CURL::libcurl
)



add_executable(shiping src/shiping.cpp)
ament_target_dependencies(shiping rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(shiping
  ${MVS_LIB}
  Threads::Threads
  dl
  ${OpenCV_LIBRARIES}
  CURL::libcurl
)






add_executable(gr src/gr.cpp)
ament_target_dependencies(gr rclcpp sensor_msgs image_transport cv_bridge OpenCV std_srvs )
target_link_libraries(gr
  ${MVS_LIB}
  Threads::Threads
  dl
  ${OpenCV_LIBRARIES}
  CURL::libcurl
)



# 如不想依赖 LD_LIBRARY_PATH，可开启 RPATH（可选）
# set_target_properties(se PROPERTIES
#   BUILD_RPATH "${MVS_LIB_PATH}"
#   INSTALL_RPATH "${MVS_LIB_PATH}"
# )

install(TARGETS
  se
 
  shiping
  

  gr
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
