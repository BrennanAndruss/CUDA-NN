#pragma once

#include "layer.h"

namespace nn 
{
    class Linear : public Layer
    {
    public:
        Linear(int inSize, int outSize);

        Tensor forward(Tensor &in) override;
        Tensor backward(Tensor &gradOut) override;

        std::vector<Tensor*> getParams() override;

        void save(std::ostream &out) const override;

        Tensor weights;
        Tensor biases;

    private:
        Tensor activationsPrev;
        Tensor zValues;

        dim3 gridDim;
    };
} // namespace nn
