#pragma once

#include "layer.h"

class Sigmoid : public Layer
{
public:
    Sigmoid(int size);

    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &gradOut) override;

    Tensor activations;

private:
    int gridSize;
};
