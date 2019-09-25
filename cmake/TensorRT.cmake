find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -O0 --keep --source-in-ptx)
endif ()

# Default to highest possible real and virtual arch (6.2 at time of this writing)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_ARCH -gencode arch=compute_62,code=sm_62) 
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH};-std=c++11")
cuda_include_directories( # Set the include dirs to pass to nvcc
    ${PYTHON27_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIR}
    )

set(TENSORRT_SOURCE_FILES src/TRTEngine.cpp)
set(TENSORRT_HEADER_FILES include/TRTEngine.hpp)
set(WRAPPER_SOURCE_FILES src/PyTensorRT.cpp)

# Set nvcc build params and create build targets
cuda_add_library(PyTensorRT SHARED ${TENSORRT_SOURCE_FILES} ${TENSORRT_HEADER_FILES} ${WRAPPER_SOURCE_FILES} )
target_link_libraries(PyTensorRT ${OpenCV_LIBS} nvinfer nvparsers)
set_target_properties(PyTensorRT PROPERTIES SUFFIX ".so" PREFIX "")

add_executable(uffToPlan src/uffToPlan.cpp)
target_link_libraries(uffToPlan nvinfer nvparsers)
