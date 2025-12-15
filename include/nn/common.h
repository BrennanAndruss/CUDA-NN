#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace nn 
{
    // Shared enums for network states
    enum class Mode
    {
        Train,
        Eval
    };

    // Error checking macro for CUDA calls
    #define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
    inline void checkCuda(cudaError_t result, const char *func, const char *file, int line)
    {
        if (result != cudaSuccess)
        {
            std::cerr << "Cuda Runtime Error at " << file << ":" << line << " in " << func << " : " 
                      << cudaGetErrorString(result) << std::endl;
            std::abort();
        }
    }

    // Constants and utility functions for CUDA kernels
    constexpr int TILE_SIZE = 16;
    constexpr int BLOCK_SIZE = TILE_SIZE * TILE_SIZE;
    constexpr dim3 BLOCK_DIM(TILE_SIZE, TILE_SIZE);

    constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }
}