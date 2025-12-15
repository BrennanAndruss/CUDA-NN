#include "nn/linear.h"
#include "nn/common.h"

namespace nn {

__global__
void forwardLinear(const float *a, const float *W, const float *b, float *z, int inSize, int outSize)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Allocate tiles in shared memory
    __shared__ float s_W[TILE_SIZE][TILE_SIZE];
    __shared__ float s_x[TILE_SIZE];

    if (row < outSize && col < inSize)
    {
        // Compute weighted sum
        float sum = 0.0f;
        for (int p = 0; p < inSize; p += TILE_SIZE)
        {
            // Collaboratively load vectors into shared memory
            s_W[ty][tx] = W[row * inSize + (p + tx)];
            if (tx == 0)
            {
                s_x[ty] = a[p + ty];
            }
            __syncthreads();

            // Dot product between row of W and tile of x
            for (int k = 0; k < TILE_SIZE; k++)
            {
                sum += s_W[ty][k] * s_x[k];
            }
            __syncthreads();
        }

        // Add bias to weighted sum
        if (col == 0)
        {
            z[row] = sum + b[row];
        }
    } 
}

__global__
void backwardLinear(const float *dz, const float *aPrev, const float *W,
    float *dW, float *db, float *daPrev, int inSize, int outSize)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Allocate tiles in shared memory
    __shared__ float s_WT[TILE_SIZE][TILE_SIZE];
    __shared__ float s_dz[TILE_SIZE];
    
    if (row < outSize && col < inSize)
    {
        // Compute aPrev gradient
        float gradaPrev = 0.0f;
        for (int p = 0; p < outSize; p += TILE_SIZE)
        {
            // Collaboratively load vectors into shared memory
            s_WT[ty][tx] = W[(p + ty) * inSize + col];
            if (tx == 0)
            {
                s_dz[ty] = dz[p + ty];
            }
            __syncthreads();

            // Dot product row of W^T and tile of dz
            // row of W^T corresponds to column of W
            for (int k = 0; k < TILE_SIZE; k++)
            {
                gradaPrev += s_WT[ty][k] * s_dz[k];
            }
        }

        daPrev[col] = gradaPrev;

        // Compute weight and bias gradients
        dW[row * inSize + col] = dz[row] * aPrev[col];
        if (col == 0)
        {
            db[row] = dz[row];
        }
    }
}

Linear::Linear(int inSize, int outSize) :
    Layer(inSize, outSize),
    weights({outSize, inSize}), biases({outSize}),
    activationsPrev({inSize}), zValues({outSize}),
    gridDim(CEIL_DIV(inSize, TILE_SIZE), CEIL_DIV(outSize, TILE_SIZE))
{
    weights.allocDevice();
    biases.allocDevice();
    zValues.allocDevice();

    weights.allocGrad();
    biases.allocGrad();
}

Tensor Linear::forward(Tensor &in)
{
    activationsPrev = in;

    forwardLinear<<<gridDim, BLOCK_DIM>>>(
        activationsPrev.data(), weights.data(), biases.data(), zValues.data(), inSize, outSize
    );

    return zValues;
}

Tensor Linear::backward(Tensor &gradOut)
{
    Tensor gradIn({inSize});
    gradIn.allocGrad();

    // Compute gradients, storing gradients for weights and biases
    backwardLinear<<<gridDim, BLOCK_DIM>>>(
        gradOut.grad(), activationsPrev.data(), weights.data(),
        weights.grad(), biases.grad(), gradIn.grad(), inSize, outSize
    );

    // Return activation gradient to propagate to previous input layer
    return gradIn;
}

std::vector<Tensor*> Linear::getParams()
{
    return std::vector<Tensor*>{&weights, &biases};
}

void Linear::save(std::ostream &out) const
{
    out << "Linear\n";
    out << inSize << " " << outSize << "\n";
    weights.save(out);
    biases.save(out);
}

} // namespace nn
