cmake_minimum_required(VERSION 3.19)
project(lab3)

set(CMAKE_CXX_STANDARD 14)
# Add executables

# Find MPI
find_package(MPI REQUIRED)

add_executable(lab3 main.cpp cmake-build-debug/one.h)

# Link against MPI
target_link_libraries(lab3 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
    set_target_properties(lab3 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(lab3 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()