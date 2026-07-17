# sam3cpplib -- SAM3 single-image promptable segmentation library.
# Works standalone (top-level) and as a git submodule of a larger app
# (add_subdirectory). All paths are CMAKE_CURRENT_SOURCE_DIR-relative so the
# folder is relocatable; the only permitted absolute paths are /usr/local/cuda*
# (found via find_package(CUDAToolkit)).
cmake_minimum_required(VERSION 3.18)

# Diamond-inclusion guard: two consumers may add_subdirectory() this library.
# Use a target check rather than a CACHE sentinel: cached sentinels survive a
# later configure and used to make this whole file return before defining the
# library.
if(TARGET sam3cpp)
    return()
endif()

project(sam3cpplib VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(SAM3CPP_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SAM3CPP_TOP_LEVEL ON)
endif()

# ── Options ─────────────────────────────────────────────────────────────────
option(SAM3CPP_CUDA          "Enable the ggml CUDA backend"                        ON)
option(SAM3CPP_TENSORRT      "Build the TensorRT backend (run scripts/setup/setup_tensorrt.sh once first)" OFF)
option(SAM3CPP_IMAGE_IO      "Compile the stb-backed sam3_load_image/sam3_save_mask helpers" ON)
option(SAM3CPP_BUILD_EXAMPLES "Build example executables"  ${SAM3CPP_TOP_LEVEL})
option(SAM3CPP_BUILD_TESTS    "Build test executables"     ${SAM3CPP_TOP_LEVEL})

# ── ggml (submodule; the CPU backend is always in -- it is the golden-sample
#    reference and the weight container's dequant layer) ─────────────────────
if(SAM3CPP_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(NOT TARGET ggml)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ggml ggml EXCLUDE_FROM_ALL)
endif()

# ── Library ─────────────────────────────────────────────────────────────────
set(SAM3CPP_SOURCES
    src/model_load.cpp
    src/tokenizer.cpp
    src/preprocess.cpp
    src/state.cpp
    src/mask_utils.cpp
    src/api.cpp
    src/debug.cpp
    src/ggml/backend.cpp
    src/ggml/backbone.cpp
    src/ggml/pcs.cpp
    src/ggml/pvs.cpp
)
if(SAM3CPP_IMAGE_IO)
    list(APPEND SAM3CPP_SOURCES src/image_io.cpp)
endif()
if(SAM3CPP_TENSORRT)
    list(APPEND SAM3CPP_SOURCES
        src/trt/trt_engine.cpp
        src/trt/trt_runtime.cpp
        src/trt/trt_encoder.cpp
        src/trt/trt_pcs.cpp
        src/trt/trt_pvs.cpp
    )
endif()

add_library(sam3cpp STATIC ${SAM3CPP_SOURCES})
add_library(sam3cpp::sam3cpp ALIAS sam3cpp)

target_include_directories(sam3cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # Compat: lets existing consumers keep `#include "sam3.h"` when they
        # migrate from upstream sam3.cpp (canonical form is "sam3cpp/sam3.h").
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sam3cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(SAM3CPP_IMAGE_IO)
    target_include_directories(sam3cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb)
endif()
target_link_libraries(sam3cpp PUBLIC ggml)
target_compile_features(sam3cpp PUBLIC cxx_std_17)

# ── TensorRT backend ────────────────────────────────────────────────────────
if(SAM3CPP_TENSORRT)
    target_compile_definitions(sam3cpp PRIVATE SAM3_TRT_ENCODER)

    set(SAM3CPP_TRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt-include"
        CACHE PATH "TensorRT public headers (NvInfer.h etc.), from scripts/setup/setup_tensorrt.sh")
    set(SAM3CPP_TRT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt-libs"
        CACHE PATH "TensorRT runtime .so location, from scripts/setup/setup_tensorrt.sh")

    find_path(SAM3CPP_TRT_INCLUDE_PATH NvInfer.h PATHS ${SAM3CPP_TRT_INCLUDE_DIR} REQUIRED)
    find_library(SAM3CPP_TRT_NVINFER nvinfer PATHS ${SAM3CPP_TRT_LIB_DIR} REQUIRED)
    find_library(SAM3CPP_TRT_NVONNXPARSER nvonnxparser PATHS ${SAM3CPP_TRT_LIB_DIR} REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    target_include_directories(sam3cpp PRIVATE ${SAM3CPP_TRT_INCLUDE_PATH})
    target_link_libraries(sam3cpp PUBLIC ${SAM3CPP_TRT_NVINFER} ${SAM3CPP_TRT_NVONNXPARSER} CUDA::cudart)
    set_target_properties(sam3cpp PROPERTIES BUILD_RPATH "${SAM3CPP_TRT_LIB_DIR}")
endif()

# ── Examples / tests ────────────────────────────────────────────────────────
if(SAM3CPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(SAM3CPP_BUILD_TESTS)
    # Must be enabled at the top-level build directory so `ctest --test-dir
    # build` discovers tests registered by the child directory.
    enable_testing()
    add_subdirectory(tests)
endif()
