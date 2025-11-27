#include "linear.h"

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
        for (int p = 0; p < inSize / TILE_SIZE; p++)
        {
            // Collaboratively load tiles into shared memory
            s_W[ty][tx] = W[row * inSize + (p * TILE_SIZE + tx)];
            if (tx == 0)
                s_x[ty] = a[p * TILE_SIZE + ty];
            __syncthreads();

            // Dot product between row of W and tile of x
            for (int k = 0; k < TILE_SIZE; k++)
                sum += s_W[ty][k] * s_x[k];
            __syncthreads();
        }

        // Add bias to weighted sum
        z[row] = sum + b[row];
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
    __shared__ float s_dZ[TILE_SIZE];
    
    if (row < outSize && col < inSize)
    {
        // Compute aPrev gradient
        float gradaPrev = 0.0f;
        for (int p = 0; p < outSize / TILE_SIZE; p++)
        {
            // Collaboratively load tiles into shared memory, transposing W
            s_WT[ty][tx] = W[(p * TILE_SIZE + ty) * inSize + col];
            if (tx == 0)
                s_dZ[ty] = dz[p * TILE_SIZE + ty];            
            __syncthreads();

            // Dot product row of W^T and tile of dz
            for (int k = 0; k < TILE_SIZE; k++)
            {
                gradaPrev += s_WT[k][tx] * s_dZ[ty];
            }
        }

        // Compute bias and weight gradients
        dW[row * inSize + col] = dz[row] * aPrev[col];
        if (col == 0)
            db[row] = dz[row];

        // Store computed aPrev gradient
        daPrev[col] = gradaPrev;
    }
}

Linear::Linear(int inSize, int outSize) :
    Layer(inSize, outSize),
    weights({outSize, inSize}), biases({outSize}),
    activationsPrev({inSize}), zValues({outSize}),
    gridDim((inSize + TILE_SIZE - 1) / TILE_SIZE, (outSize + TILE_SIZE - 1) / TILE_SIZE) 
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
