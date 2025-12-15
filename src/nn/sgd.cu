#include "nn/sgd.h"
#include "nn/common.h"

namespace nn {

__global__
void optimizeSGD(float *params, const float *grads, float learningRate, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    params[tid] -= learningRate * grads[tid];
}

SGDOptimizer::SGDOptimizer(std::vector<Tensor*> params, float learningRate) :
    Optimizer(params, learningRate) {}

void SGDOptimizer::step()
{
    for (Tensor *param : params)
    {
        int size = param->numel();
        int gridSize = CEIL_DIV(size, BLOCK_SIZE);

        // Update parameters using SGD
        optimizeSGD<<<gridSize, BLOCK_SIZE>>>(
            param->data(), param->grad(), learningRate, size
        );
    }
}

} // namespace nn
