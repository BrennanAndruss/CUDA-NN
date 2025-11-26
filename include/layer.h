#pragma once

#include "tensor.h"

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = TILE_SIZE * TILE_SIZE;
constexpr dim3 BLOCK_DIM(TILE_SIZE, TILE_SIZE);

class Layer
{
public:
    int inSize;
    int outSize;

    Layer(int inSize, int outSize) : inSize(inSize), outSize(outSize) {}
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor &in) = 0;
    virtual Tensor backward(const Tensor &gradOut) = 0;
};
