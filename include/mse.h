#pragma once

#include "loss.h"

namespace nn 
{
    class MSELoss : public LossFunction
    {
    public:
        float lossFn(const Tensor &pred, const Tensor &target) override;
        Tensor backward() override;
    };
} // namespace nn
