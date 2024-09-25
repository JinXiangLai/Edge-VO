cmake_minimum_required(VERSION 3.10)
# ROS 的g2o会污染现版本，需要把 source ros关闭
project(edge_slam)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic) # ljx
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_BUILD_TYPE "Debug")
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ELSE()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


# set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native") # 该指令可能导致意外core dump
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

add_compile_options(-Werror=return-type)


find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(edge_slam SHARED
    src/Camera.cpp
    src/WheelCameraCalib.cpp
    src/Pose.cpp
    src/Utils.cpp
    src/Landmark.cpp
    src/KeyFrame.cpp
    src/Optimizer.cpp
)
target_link_libraries(edge_slam
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
)

add_executable(main main_loop_closure_confirm.cpp)
target_link_libraries(main 
    edge_slam
)

add_executable(edge_slam_main main_edge_slam.cpp)
target_link_libraries(edge_slam_main
    edge_slam
    -lpthread
)

add_executable(ut unit_test.cpp)
target_link_libraries(ut 
    edge_slam
    -lpthread
)