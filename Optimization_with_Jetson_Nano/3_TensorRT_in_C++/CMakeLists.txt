# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(trt_yolo_cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT 라이브러리 검색 (Jetson Nano 기본 경로)
find_library(NVINFER_LIB nvinfer     PATHS /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib/aarch64-linux-gnu)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
)

add_executable(trt_yolo main.cpp)

target_link_libraries(trt_yolo
    ${OpenCV_LIBS}
    ${CUDA_CUDART_LIBRARY}
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    dl
)
