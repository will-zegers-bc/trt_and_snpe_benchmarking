set(SNPE_VERSION 1.25.1.310)
set(SNPE_ROOT $ENV{HOME}/snpe-${SNPE_VERSION})

include_directories(
    ${SNPE_ROOT}/include/zdl
)

option(AARCH64 "Build of aarch64 architecture" OFF)
if(AARCH64)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/aarch64-linux-gcc4.9")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/aarch64-linux-gcc4.9")
else()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/x86_64-linux-clang")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/x86_64-linux-clang")
endif()

se(SNPE_SOURCE_FILES src/CheckRuntime.cpp
                      src/LoadContainer.cpp
                      src/LoadInputTensor.cpp
                      src/NV21Load.cpp
                      src/PreprocessInput.cpp
                      src/SetBuilderOptions.cpp
                      src/SNPEEngine.cpp
                      src/Util.cpp
)
set(SNPE_HEADER_FILES include/CheckRuntime.hpp
                      include/LoadContainer.hpp
                      include/LoadInputTensor.hpp
                      include/NV21Load.hpp
                      include/PreprocessInput.hpp
                      include/SetBuilderOptions.hpp
                      include/SNPEEngine.hpp
                      include/Util.hpp
)
set(SNPE_WRAPPER_FILES src/PySNPE.cpp)

add_library(snpe SHARED ${SNPE_SOURCE_FILES} 
                        ${SNPE_HEADER_FILES}
                        ${SNPE_WRAPPER_FILES}
)
target_link_libraries(snpe ${PYTHON27_LIBRARIES} SNPE libsymphony-cpu.so)
set_target_properties(snpe PROPERTIES SUFFIX ".so" PREFIX "")

add_executable(snpe-sample ${SNPE_HEADER_FILES} ${SNPE_SOURCE_FILES})
target_link_libraries(snpe-sample SNPE libsymphony-cpu.so)
