#include "nn/crossentropy.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardSoftmaxCE(const float *z, float *a, const float *target, float *loss, int size)
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
                s_data[threadIdx.x], s_data[threadIdx.x + i]);
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

    // Normalize outputs and compute loss
    if (tid < size)
    {
        // Save softmax output for backward pass
        a[tid] = e / sumExp;

        // Compute cross-entropy loss
        if (target[tid] > 0.0f)
        {
            float pred = std::fmaxf(a[tid], 1e-9f);
            atomicAdd(loss, -target[tid] * logf(pred));
        }       
    }
}

__global__
void backwardSoftmaxCE(const float *da, const float *a, float *dz, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    // Gradient of softmax combined with cross-entropy loss
    dz[tid] = a[tid] - da[tid];
}

float CrossEntropyLoss::lossFn(const Tensor &logits, const Tensor &target)
{
    // Store logits for softmax backward pass
    yPred = logits;
    yTarget = target;
    yTarget.toDevice();

    float *d_loss;
    CHECK_CUDA(cudaMalloc((void**)&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));

    int size = yPred.numel();
    int gridSize = CEIL_DIV(size, BLOCK_SIZE);

    // Compute fused softmax and cross-entropy forward pass
    forwardSoftmaxCE<<<gridSize, BLOCK_SIZE>>>(
        yPred.data(), yPred.data(), yTarget.data(), d_loss, size
    );

    float h_loss = 0.0f;
    CHECK_CUDA(cudaMemcpy(
        &h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaFree(d_loss));
    return h_loss;
}

Tensor CrossEntropyLoss::backward()
{
    Tensor gradIn(yPred.getShape());
    gradIn.allocGrad();

    int size = yPred.numel();
    int gridSize = CEIL_DIV(size, BLOCK_SIZE);

    // Compute gradient using fused softmax and cross-entropy backward pass
    backwardSoftmaxCE<<<gridSize, BLOCK_SIZE>>>(
        yTarget.data(), yPred.data(), gradIn.grad(), size
    );

    return gradIn;
}

} // namespace nn
