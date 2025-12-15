#include "nn/relu.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardReLU(const float *z, float *a, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    a[tid] = fmaxf(0.0f, z[tid]);
}

__global__
void backwardReLU(const float *dA, const float *z, float *dz, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    dz[tid] = (z[tid] > 0.0f) ? dA[tid] : 0.0f;
}

ReLU::ReLU(int size) :
    Layer(size, size),
    activations({size}),
    gridSize(CEIL_DIV(size, BLOCK_SIZE))
{
    activations.allocDevice();
}

Tensor ReLU::forward(Tensor &in)
{
    forwardReLU<<<gridSize, BLOCK_SIZE>>>(in.data(), activations.data(), inSize);
    return activations;
}

Tensor ReLU::backward(Tensor &gradOut)
{
    Tensor gradIn({inSize});
    gradIn.allocGrad();

    backwardReLU<<<gridSize, BLOCK_SIZE>>>(
        gradOut.grad(), activations.data(), gradIn.grad(), inSize
    );
    
    return gradIn;
}

std::vector<Tensor*> ReLU::getParams() { return {}; }

void ReLU::save(std::ostream &out) const
{
    out << "ReLU\n";
    out << inSize << "\n";
}

} // namespace nn
