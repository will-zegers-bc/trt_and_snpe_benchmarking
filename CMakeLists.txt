cmake_minimum_required(VERSION 3.5)
project(trt_image_classification)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Default the build type to Release
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ggdb -fPIC ${REDUCE_WARNINGS} -fmax-errors=5")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -Wall -fPIC ${REDUCE_WARNINGS} -fmax-errors=5")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -O0 --keep --source-in-ptx)
endif ()

# Default to highest possible real and virtual arch (6.2 at time of this writing)
set(CUDA_ARCH -gencode arch=compute_62,code=sm_62) 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

##############################################################################
# Build
##############################################################################
set(PYBIND_DIR dep/pybind11)
if(NOT TARGET pybind11::pybind11)  # Needed when we are building standalone
    add_subdirectory(${PYBIND_DIR} pybind11)
endif()

##############################################################################
# Build target library

# Set the include dirs to pass to nvcc
set(PYTHON27_INCLUDE_DIRS /usr/include/python2.7)
set(PYBIND11_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${PYBIND_DIR}/include)
cuda_include_directories(
    ${PYTHON27_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    )

include_directories(
    ${PYTHON27_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

# CMake versions prior to v3.8 don't propogate the CMAKE_CXX_STANDARD variable to nvcc :(
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH};-std=c++11")
set(TENSORRT_SOURCE_FILES src/trt_inference_engine.cpp)
set(TENSORRT_HEADER_FILES include/trt_inference_engine.hpp)
set(WRAPPER_SOURCE_FILES src/pybind11_trt.cpp)

# Set the include dirs to pass to nvcc
# Set nvcc build params and create build targets
cuda_add_library(tensor_rt SHARED ${TENSORRT_SOURCE_FILES} ${TENSORRT_HEADER_FILES} ${WRAPPER_SOURCE_FILES} )
target_link_libraries(tensor_rt ${PYTHON27_LIBRARIES} ${OpenCV_LIBS} nvinfer nvparsers)
set_target_properties(tensor_rt PROPERTIES SUFFIX ".so" PREFIX "")

add_executable(uff_to_plan src/uff_to_plan.cpp)
target_link_libraries(uff_to_plan nvinfer nvparsers)
