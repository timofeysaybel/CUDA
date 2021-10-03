cmake_minimum_required(VERSION 3.18)
project(BasicImageConvolution CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

set(SOURCE_FILES src/image/Image.cpp src/io/Args.cpp src/io/InputException.cpp src/filters/Solver.cu src/io/Reader.cpp src/io/Writer.cpp src/main.cu)

add_executable(BasicImageConvolution ${SOURCE_FILES})

set_target_properties(BasicImageConvolution PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
