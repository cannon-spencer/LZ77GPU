cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(LZ77GPU LANGUAGES CXX CUDA)

# Modern way to find and configure CUDA
find_package(CUDAToolkit REQUIRED)

# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Add main executable and link to libcubwt and CLI11
add_executable(lz77_gpu main.cu libcubwt.cu
        main.cu)

# Fetch CLI11
include(FetchContent)
FetchContent_Declare(
        cli11
        GIT_REPOSITORY "https://github.com/CLIUtils/CLI11.git"
        GIT_TAG "v1.9.0"
)
FetchContent_MakeAvailable(cli11)

# Link CLI11 library to the target
target_link_libraries(lz77_gpu ${CUDA_LIBRARIES} CLI11::CLI11)

set_target_properties(lz77_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(lz77_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>)

target_compile_definitions(lz77_gpu PRIVATE USE_CUDA)