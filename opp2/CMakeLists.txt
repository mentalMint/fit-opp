cmake_minimum_required(VERSION 3.19)
project(opp2)

set(CMAKE_CXX_STANDARD 14)

add_executable(opp2 main.cpp vector_operations.c vector_operations.h vector_operations_omp_union.c
        vector_operations_omp_union.h double.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")
