#!/bin/bash
# 针对Quadro P2200的优化设置

# 禁用硬件加速相关的问题
export LIBGL_ALWAYS_SOFTWARE=0
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=0

# NVIDIA Quadro专业驱动优化
export __GL_SYNC_TO_VBLANK=0
export __GL_THREADED_OPTIMIZATIONS=0
export __GL_YIELD="USLEEP"

# VTK优化设置
export VTK_USE_DIRECTX=0
export VTK_USE_OPENGL=1
export VTK_OPENGL_HAS_OSMESA=0

# 日志设置
export VTK_LOGGING_VERBOSITY=WARNING

# 运行程序
echo "Running with Quadro P2200 optimizations..."
./rgbd_slam "$@"