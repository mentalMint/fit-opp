cmake_minimum_required(VERSION 3.21)
project(lab4)

set(CMAKE_CXX_STANDARD 17)

add_executable(lab4 main.cpp )
#add_executable(lab4 createBoard.cpp)

find_package(MPI REQUIRED)

# Link against MPI
target_link_libraries(lab4 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
