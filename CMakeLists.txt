###############################################################################
#  Top-level CMake for LZ77GPU
###############################################################################
cmake_minimum_required(VERSION 3.18)
project(LZ77GPU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

###############################################################################
#  Helper – just for IDE grouping
###############################################################################
macro(group_sources BASE_DIR)
    if (MSVC)
        file(GLOB_RECURSE ALL_SOURCE_FILES CONFIGURE_DEPENDS ${BASE_DIR}/*)
        foreach(SOURCE_FILE ${ALL_SOURCE_FILES})
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
set(UTIL_SOURCES src/Util/src/timer.cu)

set(SA_SOURCES  src/SuffixArray/src/prefix_doubling.cu)
set(SA_MAIN     src/SuffixArray/src/sa_profiling_main.cu)

set(LZ77_SOURCES src/LZ77/src/kernel.cu
        src/LZ77/src/processor.cu)
set(LZ77_MAIN    src/LZ77/src/LZ77_main.cu)

set(TEST_SOURCES src/Tests/lz77_tests.cu)

group_sources(${CMAKE_CURRENT_SOURCE_DIR}/src)

###############################################################################
#  Third-party deps  (FetchContent)
###############################################################################
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP QUIET)

# ── SDSL ─────────────────────────────────────────────────────────────────────
FetchContent_Declare(sdsl
        GIT_REPOSITORY https://github.com/simongog/sdsl-lite.git)
set(GENERATE_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sdsl)

# ── divsufsort ───────────────────────────────────────────────────────────────
FetchContent_Declare(divsufsort
        GIT_REPOSITORY https://github.com/simongog/libdivsufsort.git
        GIT_TAG        2.0.1)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(BUILD_DIVSUFSORT64 ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(divsufsort)
set(DIVSUFSORT_INCLUDE_DIR ${divsufsort_SOURCE_DIR}/include)

# ── CLI11 ────────────────────────────────────────────────────────────────────
FetchContent_Declare(cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v1.9.0)
FetchContent_MakeAvailable(cli11)

# ── malloc_count ─────────────────────────────────────────────────────────────
FetchContent_Declare(malloc_count
        GIT_REPOSITORY https://github.com/bingmann/malloc_count.git
        GIT_TAG        v0.7.1)
FetchContent_MakeAvailable(malloc_count)
add_library(malloc_count STATIC ${malloc_count_SOURCE_DIR}/malloc_count.c)
target_include_directories(malloc_count PUBLIC ${malloc_count_SOURCE_DIR})

# --- Catch2 (unit-test framework) -------------------------------------------
FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

# --- Enable Catch2 test discovery ------------------------------------------
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)


###############################################################################
#  Static lib for libcubwt.cu
###############################################################################
add_library(cubwt STATIC src/SuffixArray/src/libcubwt.cu)
target_include_directories(cubwt PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/src)
set_target_properties(cubwt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

###############################################################################
#  Utility lib
###############################################################################
add_library(util STATIC ${UTIL_SOURCES} ${UTIL_HEADERS})
target_include_directories(util PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Util/include
        ${DIVSUFSORT_INCLUDE_DIR})
target_link_libraries(util PUBLIC CUDA::cudart)


###############################################################################
#  Suffix Array static library
###############################################################################
add_library(suffix_array STATIC
        src/SuffixArray/src/prefix_doubling.cu)

target_include_directories(suffix_array PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SuffixArray/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Util/include)

set_target_properties(suffix_array PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

###############################################################################
#  Main executables
###############################################################################
add_executable(sa_profiler ${SA_MAIN})
add_executable(lz77        ${LZ77_MAIN} ${LZ77_SOURCES})

###############################################################################
#  Shared include paths & options
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
            $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math --extended-lambda --expt-relaxed-constexpr>)
    set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

###############################################################################
#  Unit-tests (Catch2 + CTest)
###############################################################################
add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE ${ALL_INCLUDES})
target_link_libraries(tests PRIVATE
        Catch2::Catch2WithMain
        cubwt util malloc_count
        CLI11::CLI11
        sdsl divsufsort divsufsort64
        Threads::Threads
        $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>)

catch_discover_tests(tests)


# --- Link suffix array static library where used ------------------------------------------

# link sa_profiler
target_link_libraries(sa_profiler PRIVATE suffix_array)

# link lz77
target_link_libraries(lz77 PRIVATE suffix_array)

# link tests
target_link_libraries(tests PRIVATE suffix_array)

