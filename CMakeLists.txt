cmake_minimum_required(VERSION 3.16)
project(opencv_vkstereo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VKSTEREO_BUILD_TESTS "Build the GoogleTest suite" ON)
option(VKSTEREO_BUILD_SAMPLES "Build the benchmark sample" ON)

find_package(OpenCV REQUIRED COMPONENTS core calib3d imgcodecs imgproc)
find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

add_subdirectory(modules/vkstereo)
