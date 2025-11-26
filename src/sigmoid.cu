#include "sigmoid.h"

__global__
void forwardSigmoid(const float *Z, float *a, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    a[tid] = 1.0f / (1.0f + expf(-Z[tid]));
}

__global__
void backwardSigmoid(const float *da, const float *a, float *dZ, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float sigmoid = a[tid];
    dZ[tid] = da[tid] * sigmoid * (1.0f - sigmoid);
}

Sigmoid::Sigmoid(int size) :
    Layer(size, size),
    activations(size),
    gridSize((size + BLOCK_SIZE - 1) / BLOCK_SIZE)
{
    activations.allocDevice();
}

Tensor Sigmoid::forward(const Tensor &in)
{
    forwardSigmoid<<<gridSize, BLOCK_SIZE>>>(in.data(), activations.data(), inSize);
    return activations;
}

Tensor Sigmoid::backward(const Tensor &gradOut)
{
    Tensor gradIn(inSize);
    gradIn.allocDevice();

    backwardSigmoid<<<gridSize, BLOCK_SIZE>>>(
        gradOut.data(), activations.data(), gradIn.data(), inSize
    );
    
    return gradIn;
}
