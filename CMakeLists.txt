cmake_minimum_required(VERSION 3.22.1)
project(UnisocCompute)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(unisoc_compute SHARED native-lib.cpp)

find_library(log-lib log)
find_library(dl-lib dl)

target_link_libraries(unisoc_compute ${log-lib} ${dl-lib})

target_compile_options(unisoc_compute PRIVATE -O3 -funsafe-math-optimizations)
