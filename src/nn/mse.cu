#include "nn/mse.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardMSE(const float *pred, const float *target, float *loss, int size)
{
    __shared__ float s_loss[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    // Compute squared errors
    float diff = pred[tid] - target[tid];
    s_loss[threadIdx.x] = diff * diff;
    __syncthreads();

    // Reduce within block
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i && (tid + i) < size)
        {
            s_loss[threadIdx.x] += s_loss[threadIdx.x + i];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (threadIdx.x == 0)
    {
        atomicAdd(loss, s_loss[0]);
    }
}

__global__
void backwardMSE(const float *pred, const float *target, float *gradIn, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    // Compute gradient of MSE loss
    gradIn[tid] = 2.0f * (pred[tid] - target[tid]) / size;
}

float MSELoss::lossFn(const Tensor &pred, const Tensor &target)
{
    yPred = pred;
    yTarget = target;
    yTarget.toDevice();

    float *d_loss;
    CHECK_CUDA(cudaMalloc((void**)&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));

    int size = pred.numel();
    int gridSize = CEIL_DIV(size, BLOCK_SIZE);

    // Compute MSE loss
    forwardMSE<<<gridSize, BLOCK_SIZE>>>(
        yPred.data(), yTarget.data(), d_loss, size
    );

    float h_loss = 0.0f;
    CHECK_CUDA(cudaMemcpy(
        &h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaFree(d_loss));
    return h_loss / size;
}

Tensor MSELoss::backward()
{
    Tensor gradIn(yPred.getShape());
    gradIn.allocGrad();

    int size = yPred.numel();
    int gridSize = CEIL_DIV(size, BLOCK_SIZE);

    // Compute gradient of MSE loss
    backwardMSE<<<gridSize, BLOCK_SIZE>>>(
        yPred.data(), yTarget.data(), gradIn.grad(), size
    );

    return gradIn;
}

} // namespace nn