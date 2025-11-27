#pragma once

#include "layer.h"

class Sigmoid : public Layer
{
public:
    Sigmoid(int size);

    Tensor forward(Tensor &input) override;
    Tensor backward(Tensor &gradOut) override;

    std::vector<Tensor*> getParams() override;

    void save(std::ostream &out) const override;

    Tensor activations;

private:
    int gridSize;
};
