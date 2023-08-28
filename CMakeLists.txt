cmake_minimum_required(VERSION 3.27)

project(MyReLu CXX CUDA)

# if you have python and pytorch installed in a non-standard location, you can use this to find them
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# source file and header
set(SOURCE_FILE 
    ${CMAKE_SOURCE_DIR}/include/utils.h
    ${CMAKE_SOURCE_DIR}/kernel/relu.cu
    ${CMAKE_SOURCE_DIR}/pytorch/relu.cpp
)

# List all your code files here
add_library(MyReLU SHARED
    ${SOURCE_FILE}
)

target_compile_features(MyReLU PRIVATE cxx_std_17)
target_include_directories(MyReLU PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(MyReLU PRIVATE ${TORCH_LIBRARIES} Python::Python ${TORCH_PYTHON_LIBRARY})

# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>")

