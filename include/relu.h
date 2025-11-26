#pragma once

#include "layer.h"

class ReLU : public Layer
{
public:
    ReLU(int size);

    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &gradOut) override;

    Tensor activations;

private:
    int gridSize;
};
