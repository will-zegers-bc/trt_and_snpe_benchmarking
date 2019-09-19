set(SNPE_VERSION 1.25.1.310)
set(SNPE_ROOT $ENV{HOME}/snpe-${SNPE_VERSION})

include_directories(${SNPE_ROOT}/include/zdl)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/x86_64-linux-clang")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L ${SNPE_ROOT}/lib/aarch64-linux-gcc4.9")
endif()

set(SNPE_SOURCE_FILES src/CheckRuntime.cpp
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
target_link_libraries(snpe  SNPE symphony-cpu.so)
set_target_properties(snpe PROPERTIES SUFFIX ".so" PREFIX "")
