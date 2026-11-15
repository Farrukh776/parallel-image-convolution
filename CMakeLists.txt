cmake_minimum_required(VERSION 3.16)
project(ParallelConvolution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Makefiles, Ninja) default to an unoptimised build,
# which would make every timing meaningless.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4)
    # MSVC flags standard C functions such as sscanf as "unsafe"; the
    # suggested replacements are Microsoft-only and would break g++ builds.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Everything except main() lives in a library so the tests can link against it.
add_library(convcore STATIC
    src/image.cpp
    src/kernels.cpp
    src/convolve.cpp
)
target_include_directories(convcore PUBLIC src)

# Adds the compiler's OpenMP flag (/openmp for MSVC, -fopenmp for g++/clang)
# and links its runtime. Without the flag the #pragma omp lines are silently
# ignored and everything runs on one thread.
find_package(OpenMP REQUIRED)
target_link_libraries(convcore PUBLIC OpenMP::OpenMP_CXX)

add_executable(convolve src/main.cpp)
target_link_libraries(convolve PRIVATE convcore)

enable_testing()
add_executable(test_convolve tests/test_convolve.cpp)
target_link_libraries(test_convolve PRIVATE convcore)
add_test(NAME test_convolve COMMAND test_convolve)
