###############################################################################
#  Top-level CMake for LZ77GPU
###############################################################################
cmake_minimum_required(VERSION 3.18)
project(LZ77GPU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD   17)
set(CMAKE_CUDA_STANDARD  17)
set(CMAKE_CUDA_ARCHITECTURES 80)

###############################################################################
#  Helper – just for IDE grouping
###############################################################################
macro(group_sources BASE_DIR)
    if (MSVC)
        file(GLOB_RECURSE ALL_SOURCE_FILES CONFIGURE_DEPENDS ${BASE_DIR}/*)
        foreach (SOURCE_FILE ${ALL_SOURCE_FILES})
            file(RELATIVE_PATH FILE_PATH_REL "${BASE_DIR}" "${SOURCE_FILE}")
            get_filename_component(FILE_PATH "${FILE_PATH_REL}" PATH)
            string(REPLACE "/" "\\\\" FILE_PATH_GROUP "${FILE_PATH}")
            source_group("${FILE_PATH_GROUP}" FILES "${SOURCE_FILE}")
        endforeach()
    endif()
endmacro()

###############################################################################
#  Source lists
###############################################################################
set(UTIL_HEADERS
        src/Util/include/cuda_utils.cuh
        src/Util/include/profiler.cuh
        src/Util/include/memory_monitor.cuh
        src/Util/include/timer.cuh)
set(UTIL_SOURCES  src/Util/src/timer.cu)

set(SA_SOURCES     src/SuffixArray/src/prefix_doubling.cu)
set(SA_MAIN        src/SuffixArray/src/sa_profiling_main.cu)

set(LZ77_SOURCES   src/LZ77/src/kernel.cu
        src/LZ77/src/processor.cu)
set(LZ77_MAIN      src/LZ77/src/LZ77_main.cu)

group_sources(${CMAKE_CURRENT_SOURCE_DIR}/src)

###############################################################################
#  Third-party deps
###############################################################################
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP          QUIET)

# SDSL
FetchContent_Declare(sdsl GIT_REPOSITORY https://github.com/simongog/sdsl-lite)
set(GENERATE_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sdsl)

# divsufsort
FetchContent_Declare(divsufsort
        GIT_REPOSITORY https://github.com/simongog/libdivsufsort.git
        GIT_TAG        2.0.1)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(BUILD_DIVSUFSORT64 ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(divsufsort)

set(DIVSUFSORT_INCLUDE_DIR ${divsufsort_SOURCE_DIR}/include)

# CLI11
FetchContent_Declare(cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v1.9.0)
FetchContent_MakeAvailable(cli11)

# malloc_count
FetchContent_Declare(malloc_count
        GIT_REPOSITORY https://github.com/bingmann/malloc_count.git
        GIT_TAG        v0.7.1)
FetchContent_MakeAvailable(malloc_count)

add_library(malloc_count STATIC ${malloc_count_SOURCE_DIR}/malloc_count.c)
target_include_directories(malloc_count PUBLIC ${malloc_count_SOURCE_DIR})

###############################################################################
#  single static library that holds libcubwt.cu
###############################################################################
add_library(cubwt STATIC src/SuffixArray/src/libcubwt.cu)
target_include_directories(cubwt PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/src)
set_target_properties(cubwt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

###############################################################################
#  Utility module (unchanged)
###############################################################################
add_library(util STATIC ${UTIL_SOURCES} ${UTIL_HEADERS})
target_include_directories(util PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Util/include
        ${DIVSUFSORT_INCLUDE_DIR})
target_link_libraries(util PUBLIC CUDA::cudart)

###############################################################################
#  Main executables
###############################################################################
add_executable(sa_profiler ${SA_MAIN} ${SA_SOURCES})
add_executable(lz77        ${LZ77_MAIN} ${LZ77_SOURCES})

###############################################################################
#  Common configuration
###############################################################################
set(ALL_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Util/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/LZ77/include
        ${DIVSUFSORT_INCLUDE_DIR})

foreach(tgt IN ITEMS sa_profiler lz77)
    target_include_directories(${tgt} PRIVATE ${ALL_INCLUDES})

    target_link_libraries(${tgt} PRIVATE
            cubwt util malloc_count
            CLI11::CLI11
            sdsl divsufsort divsufsort64
            $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>)

    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            -use_fast_math
            --extended-lambda
            --expt-relaxed-constexpr>)

    set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

###############################################################################
#  Propagate divsufsort headers
###############################################################################
target_include_directories(divsufsort   INTERFACE
        $<BUILD_INTERFACE:${divsufsort_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${divsufsort_BINARY_DIR}/include>)
target_include_directories(divsufsort64 INTERFACE
        $<BUILD_INTERFACE:${divsufsort_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${divsufsort_BINARY_DIR}/include>)

foreach(tgt IN ITEMS lz77 sa_profiler test)
    if (TARGET ${tgt})
        target_link_libraries(${tgt} PRIVATE
                sdsl divsufsort divsufsort64 malloc_count CLI11::CLI11
                Threads::Threads
                $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>)
    endif()
endforeach()
