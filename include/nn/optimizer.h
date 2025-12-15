#pragma once

#include <vector>
#include "tensor.h"

namespace nn
{
    class Optimizer
    {
    public:
        Optimizer(std::vector<Tensor*> params, float learningRate) :
            params(params), learningRate(learningRate) {}
        
        virtual ~Optimizer() = default;

        virtual void step() = 0;

    protected:
        std::vector<Tensor*> params;
        float learningRate;
    };
} // namespace nn
