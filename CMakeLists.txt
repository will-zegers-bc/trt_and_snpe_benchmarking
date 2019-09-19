cmake_minimum_required(VERSION 3.5)
project(nn_engine_benchmarking)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Default the build type to Release
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

option(TENSOR_RT "Build runtime engine modules for TensorRT" OFF)
option(SNPE "Build runtime engine modules for Qualcomm SNPE" OFF)

set(PYBIND_DIR third_party/pybind11)
if(NOT TARGET pybind11::pybind11)  # Needed when we are building standalone
    add_subdirectory(${PYBIND_DIR} pybind11)
endif()

set(PYTHON27_INCLUDE_DIRS /usr/include/python2.7)

set(PYBIND11_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${PYBIND_DIR}/include)
include_directories(
    ${PYTHON27_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ggdb -fPIC ${REDUCE_WARNINGS} -fmax-errors=5")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -Wall -fPIC ${REDUCE_WARNINGS} -fmax-errors=5")
endif()

if(TENSOR_RT)
    include(cmake/TensorRT.cmake)
endif()

if(SNPE)
    include(cmake/SNPE.cmake)
endif()
