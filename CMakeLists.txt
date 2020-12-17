cmake_minimum_required(VERSION 3.16)
project(Project CUDA)

add_executable(Project main.cu)

set_target_properties(
        Project
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)