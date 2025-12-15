#include "nn/sigmoid.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardSigmoid(const float *z, float *a, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    a[tid] = 1.0f / (1.0f + expf(-z[tid]));
}

__global__
void backwardSigmoid(const float *da, const float *a, float *dz, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float sigmoid = a[tid];
    dz[tid] = da[tid] * sigmoid * (1.0f - sigmoid);
}

Sigmoid::Sigmoid(int size) :
    Layer(size, size),
    activations({size}),
    gridSize(CEIL_DIV(size, BLOCK_SIZE))
{
    activations.allocDevice();
}

Tensor Sigmoid::forward(Tensor &in)
{
    forwardSigmoid<<<gridSize, BLOCK_SIZE>>>(in.data(), activations.data(), inSize);
    return activations;
}

Tensor Sigmoid::backward(Tensor &gradOut)
{
    Tensor gradIn({inSize});
    gradIn.allocGrad();

    backwardSigmoid<<<gridSize, BLOCK_SIZE>>>(
        gradOut.grad(), activations.data(), gradIn.grad(), inSize
    );
    
    return gradIn;
}

std::vector<Tensor*> Sigmoid::getParams() { return {}; }

void Sigmoid::save(std::ostream &out) const
{
    out << "Sigmoid\n";
    out << inSize << "\n";
}

} // namespace nn
