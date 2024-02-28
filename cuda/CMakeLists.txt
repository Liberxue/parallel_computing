cmake_minimum_required(VERSION 3.27)
project(cuda)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(hello)

add_executable(cuda
        abs/main.cu
        count_bits/main.cu
        count_ones/main.cu
        fibonacci/main.cu
        find_first_zero_byte/main.cu
        find_lsb/main.cu
        find_msb/main.cu
        hello/Kernel.cu
        hello/Kernel.h
        hello/vector_addition.cpp
        hello_cuda/check_dim.cu
        hello_cuda/hello_cuda.cu
        hello_cuda/vector_sum.cu
        memory_management/vector_multiple_threads.cu
        reverse_bits/main.cu
        swap_odd_even_bits/main.cu
        three_sum/main.cu
        two_sum/main.cu
        vector_addition/vector.cu
        vector_addition/vector_multiple_threads.cu
        xor/main.cu)
