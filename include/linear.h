#pragma once

#include "layer.h"

class Linear : public Layer
{
public:
    Linear(int inSize, int outSize);

    Tensor forward(const Tensor &in) override;
    Tensor backward(const Tensor &gradOut) override;

    Tensor weights;
    Tensor biases;

private:
    Tensor activationsPrev;
    Tensor zValues;

    dim3 gridDim;
};
