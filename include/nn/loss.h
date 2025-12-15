#pragma once

#include "tensor.h"

namespace nn
{
    class LossFunction
    {
    public:
        virtual ~LossFunction() = default;

        virtual float lossFn(const Tensor &pred, const Tensor &target) = 0;
        virtual Tensor backward() = 0;

    protected:
        Tensor yPred, yTarget;
    };
} // namespace nn
