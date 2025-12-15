#pragma once

#include "optimizer.h"

namespace nn 
{
    class SGDOptimizer : public Optimizer 
    {
    public:
        SGDOptimizer(std::vector<Tensor*> params, float learningRate);

        void step() override;
    };
} // namespace nn
