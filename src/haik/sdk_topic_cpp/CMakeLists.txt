cmake_minimum_required(VERSION 3.8)
project(sdk_topic_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# ==== 海康 MVS SDK 路径（可用环境变量覆盖）====
# 例如：export HIK_SDK_ROOT=/opt/MVS
if(NOT DEFINED ENV{HIK_SDK_ROOT})
  set(ENV{HIK_SDK_ROOT} "/opt/MVS")
endif()
set(HIK_SDK_ROOT $ENV{HIK_SDK_ROOT})

# 头文件目录
include_directories(${HIK_SDK_ROOT}/include)

# 指定库的绝对路径（最稳）
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

# 可执行文件
add_executable(se src/se.cpp)
ament_target_dependencies(se rclcpp sensor_msgs)
target_link_libraries(se
  ${MVS_LIB}
  Threads::Threads
  dl
)

# 如不想依赖 LD_LIBRARY_PATH，可打开 RPATH（可选）
# set_target_properties(se PROPERTIES
#   BUILD_RPATH "${MVS_LIB_PATH}"
#   INSTALL_RPATH "${MVS_LIB_PATH}"
# )

install(TARGETS
  se
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
