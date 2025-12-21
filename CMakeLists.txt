cmake_minimum_required(VERSION 3.10)
# ROS 的g2o会污染现版本，需要把 source ros关闭
project(edge_slam)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# 编译器警告（仅对支持 GCC/Clang 生效）
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 构建类型优化
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 -march=native -ffast-math -funroll-loops -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -march=native -ffast-math -funroll-loops -DNDEBUG")
  #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 -g")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -g")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native") # 该指令可能导致意外core dump
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

add_compile_options(-Werror=return-type)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)

# 添加一些debug 宏配置
# add_definitions(-DWRITE_MATCH_PAIR_IMAGE)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

#add_definitions(-DUSE_SUPERPOINT_AND_LIGHTGLUE)
add_definitions(-DUSE_CERES2)
find_package(CUDA REQUIRED)

find_package(Ceres 2 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Ceres_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/3rdparty/tensorrtbuffer/include
)

add_library(edge_slam SHARED
    src/Config.cpp
    src/Camera.cpp
    src/WheelCameraCalib.cpp
    src/Pose.cpp
    src/Utils.cpp
    src/Landmark.cpp
    src/KeyFrame.cpp
    src/Optimizer.cpp
    src/Initializer.cpp
    src/SuperPoint.cpp
    src/LightGlue.cpp
    src/OpticalFlowStruct.cpp
    src/ceres_problem.cpp
)
target_link_libraries(edge_slam
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    -lyaml-cpp
    -lfmt
    Ceres::ceres # 若是使用ceres会与opencv的冲突，所以必须要命名空间，这里的命名空间就是存放libceres.a的文件夹名字

    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    tensorrtbuffer
)

add_executable(rgbd_slam main_rgbd_slam.cpp)
target_link_libraries(rgbd_slam
    edge_slam
    -lpthread
    -lyaml-cpp
    -lfmt
)

add_executable(sp_lg_sequence sp_lg_sequence.cpp)
target_link_libraries(sp_lg_sequence
    edge_slam
    -lpthread
    -lyaml-cpp
    -lfmt
)

add_executable(loop_closure_main loop_closure_main.cpp)
target_link_libraries(loop_closure_main
    edge_slam
    -lpthread
    -lyaml-cpp
    -lfmt
)

# add_executable(ut unit_test.cpp)
# target_link_libraries(ut 
#     edge_slam
#     -lpthread
#     -lyaml-cpp
# )

# add_executable(orb varify_ORB_orienttaion_invariance.cpp)
# target_link_libraries(orb 
#     ${OpenCV_LIBS}
#     ${EIGEN3_LIBS}
# )