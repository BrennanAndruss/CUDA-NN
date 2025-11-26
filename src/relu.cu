#include "relu.h"

__global__
void forwardReLU(const float *Z, float *a, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    a[tid] = fmaxf(0.0f, Z[tid]);
}

__global__
void backwardReLU(const float *dA, const float *Z, float *dZ, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    dZ[tid] = (Z[tid] > 0.0f) ? dA[tid] : 0.0f;
}

ReLU::ReLU(int size) :
    Layer(size, size),
    activations(size),
    gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE)
{
    activations.allocDevice();
}

Tensor ReLU::forward(const Tensor &in)
{
    forwardReLU<<<gridSize, BLOCK_SIZE>>>(in.data(), activations.data(), inSize);
    return activations;
}

Tensor ReLU::backward(const Tensor &gradOut)
{
    Tensor gradIn(inSize);
    gradIn.allocDevice();

    backwardReLU<<<gridSize, BLOCK_SIZE>>>(
        gradOut.data(), activations.data(), gradIn.data(), inSize
    );
    
    return gradIn;
}