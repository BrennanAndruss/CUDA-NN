#include "nn/softmax.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardSoftmax(const float *z, float *a, int size)
{
    __shared__ float s_data[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    s_data[threadIdx.x] = (tid < size) ? z[tid] : -FLT_MAX;
    __syncthreads();

    // Reduce to find global max
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i && (tid + i) < size)
        {
            s_data[threadIdx.x] = std::fmaxf(
                s_data[threadIdx.x], s_data[threadIdx.x + i]
            );
        }
        __syncthreads();
    }
    float maxVal = s_data[0];
    __syncthreads();

    // Calculate exponentials
    float e = (tid < size) ? expf(z[tid] - maxVal) : 0.0f;
    s_data[threadIdx.x] = e;
    __syncthreads();

    // Reduce to compute sum
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i && (tid + i) < size)
        {
            s_data[threadIdx.x] += s_data[threadIdx.x + i];
        }
        __syncthreads();
    }
    float sumExp = s_data[0];
    __syncthreads();

    // Normalize the outputs
    if (tid < size)
    {
        a[tid] = e / sumExp;
    }
}

Softmax::Softmax(int size) : Layer(size, size), 
    gridSize(CEIL_DIV(size, BLOCK_SIZE)) {}

Tensor Softmax::forward(Tensor &in)
{
    // Softmax is fused with cross-entropy loss during training
    if (mode == Mode::Train)
    {
        // Passthrough logits
        return in;
    }

    Tensor activations(in.getShape());
    activations.allocDevice();
    
    forwardSoftmax<<<gridSize, BLOCK_SIZE>>>(
        in.data(), activations.data(), inSize
    );
    return activations;
}

Tensor Softmax::backward(Tensor &gradOut)
{
    // Softmax is fused with cross-entropy loss during backpropagation
    return gradOut;
}

std::vector<Tensor*> Softmax::getParams() { return {}; }

void Softmax::save(std::ostream &out) const
{
    out << "Softmax\n";
    out << inSize << "\n";
}

} // namespace nn
