cmake_minimum_required(VERSION 3.21)
project(lab5)

set(CMAKE_CXX_STANDARD 14)

add_executable(lab5 main.cpp Task.h)

find_package(MPI REQUIRED)

# Link against MPI
target_link_libraries(lab5 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})

