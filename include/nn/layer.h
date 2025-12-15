#pragma once

#include <vector>
#include "tensor.h"

namespace nn 
{
    class Layer
    {
    public:
        int inSize;
        int outSize;

        Layer(int inSize, int outSize) : inSize(inSize), outSize(outSize) {}
        virtual ~Layer() = default;

        virtual Tensor forward(Tensor &in) = 0;
        virtual Tensor backward(Tensor &gradOut) = 0;

        virtual std::vector<Tensor*> getParams() = 0;

        virtual void save(std::ostream &out) const = 0;
    };
} // namespace nn
