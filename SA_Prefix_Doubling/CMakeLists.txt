cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 80)
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0127 NEW)
project(LZ77GPU LANGUAGES C CXX CUDA)  # Fixed: Added "C"

# Add executables
set(PSV_NSV_SOURCES
        src/timer.cu
        src/kernel.cu
        src/processor.cu
)
add_executable(lz77_gpu main.cu libcubwt.cu ${PSV_NSV_SOURCES})

add_executable(test test.cu libcubwt.cu)

include(FetchContent)

# Modern way to find and configure CUDA
find_package(CUDAToolkit REQUIRED)

# Find OpenMP
find_package(OpenMP)

# sdsl
FetchContent_Declare(
        sdsl
        GIT_REPOSITORY https://github.com/simongog/sdsl-lite
)
FetchContent_GetProperties(sdsl)
if(NOT sdsl_POPULATED)
    FetchContent_Populate(sdsl)
    set(GENERATE_DOC OFF CACHE BOOL "Do not generate doxygen for sdsl-lite")
    add_subdirectory(${sdsl_SOURCE_DIR} ${sdsl_BINARY_DIR} EXCLUDE_FROM_ALL)
    target_include_directories(test PRIVATE ${sdsl_SOURCE_DIR}/include)
endif()

# malloc_count
FetchContent_Declare(
        malloc_count
        GIT_REPOSITORY https://github.com/bingmann/malloc_count.git
        GIT_TAG        v0.7.1
)
FetchContent_GetProperties(malloc_count)
if(NOT malloc_count_POPULATED)
    FetchContent_Populate(malloc_count)
    add_library(malloc_count STATIC
            ${malloc_count_SOURCE_DIR}/malloc_count.c
    )
    target_include_directories(malloc_count PUBLIC "${malloc_count_SOURCE_DIR}")
    target_link_libraries(malloc_count dl)
    set_target_properties(malloc_count PROPERTIES LINKER_LANGUAGE C)  # Fixed
endif()

# divsuffsort
FetchContent_Declare(
        divsufsort
        GIT_REPOSITORY https://github.com/simongog/libdivsufsort.git
        GIT_TAG        2.0.1
)
FetchContent_GetProperties(divsufsort)
if(NOT divsufsort_POPULATED)
    FetchContent_Populate(divsufsort)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Do not build a shared library for libdivsufsort")
    set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build libdivsufsort example")
    set(BUILD_DIVSUFSORT64 ON CACHE BOOL "Build libdivsufsort in 64-bits mode")
    add_subdirectory(${divsufsort_SOURCE_DIR} ${divsufsort_BINARY_DIR} EXCLUDE_FROM_ALL)
    target_include_directories(divsufsort PUBLIC "${divsufsort_BINARY_DIR}/include")
    target_include_directories(divsufsort64 PUBLIC "${divsufsort_BINARY_DIR}/include")
endif()

# Include directories
include_directories(BEFORE
        ${CUDAToolkit_INCLUDE_DIRS}
        include
        ${SDSL_INCLUDE_PATH}
)

# Fetch CLI11
FetchContent_Declare(
        cli11
        GIT_REPOSITORY "https://github.com/CLIUtils/CLI11.git"
        GIT_TAG "v1.9.0"
)
FetchContent_MakeAvailable(cli11)

# Link CLI11 library to the target
target_link_libraries(lz77_gpu
        ${CUDA_LIBRARIES}
        CLI11::CLI11
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        malloc_count  # Fixed: Now linked properly
        sdsl divsufsort divsufsort64
)

set_target_properties(lz77_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(lz77_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Wall
        -Xcompiler ${OpenMP_CXX_FLAGS}
        -use_fast_math
        --extended-lambda
        --expt-relaxed-constexpr
        >
)

target_compile_definitions(lz77_gpu PRIVATE USE_CUDA)

target_include_directories(lz77_gpu PRIVATE include)

target_link_libraries(test
        ${CUDA_LIBRARIES}
        CLI11::CLI11
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
        malloc_count  # Fixed
        sdsl divsufsort divsufsort64
)

target_compile_options(test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Wall
        -Xcompiler ${OpenMP_CXX_FLAGS}
        -use_fast_math
        --extended-lambda
        --expt-relaxed-constexpr
        >
)

target_compile_definitions(test PRIVATE USE_CUDA)
target_include_directories(test PRIVATE include)
